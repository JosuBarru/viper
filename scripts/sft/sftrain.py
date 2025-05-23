import argparse
import datetime
import logging
import os
import sys
from typing import Dict, Optional

import torch
import wandb
import datasets
import numpy as np

import matplotlib.pyplot as plt
from matplotlib.ticker import PercentFormatter


from unsloth import FastLanguageModel, is_bfloat16_supported
from transformers import (
    Trainer,
    TrainingArguments,
    AutoTokenizer,
    DataCollatorForLanguageModeling,
)

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

os.chdir("/sorgin1/users/jbarrutia006/viper")  # Adjust to your working directory if needed

def parse_args():
    parser = argparse.ArgumentParser(description="Train a model using supervised fine-tuning on a QA dataset")

    parser.add_argument("--model_name", type=str, default="meta-llama/Meta-Llama-3-8B",
                        help="Pretrained model name or path")

    # SFT datasets: question-answer format
    parser.add_argument("--train_dataset_sft", type=str, required=True,
                        help="Path to the *SFT* train dataset (question-answer pairs).")
    parser.add_argument("--dev_dataset_sft", type=str, required=True,
                        help="Path to the *SFT* dev dataset (question-answer pairs).")

    # Optionally, you can also include a DPO-style dev dataset for measuring DPO loss:
    parser.add_argument("--dev_dataset_dpo", type=str, default=None,
                        help="Path to a DPO-style dev dataset if you want to compute DPO loss during SFT dev.")

    parser.add_argument("--output_dir", type=str, default="./sft_llama3",
                        help="Directory to save the trained model")

    parser.add_argument("--batch_size", type=int, default=4, help="Batch size for training")
    parser.add_argument("--gradient_accumulation", type=int, default=16, help="Gradient accumulation steps")
    parser.add_argument("--learning_rate", type=float, default=1e-5, help="Learning rate for training")
    parser.add_argument("--epochs", type=int, default=3, help="Number of training epochs")
    parser.add_argument("--max_steps", type=int, default=-1, help="Maximum number of training steps")

    parser.add_argument("--logging_steps", type=int, default=10, help="Logging frequency")
    parser.add_argument("--eval_steps", type=int, default=200, help="Evaluation frequency")
    parser.add_argument("--save_steps", type=int, help="Model saving frequency")

    parser.add_argument("--lr_scheduler_type", type=str, default="cosine",
                        help="Type of learning rate scheduler (e.g. linear, cosine)")
    parser.add_argument("--warmup_ratio", type=float, default=0.1,
                        help="Warmup ratio for learning rate scheduling")

    parser.add_argument("--weight_decay", type=float, default=0.01, help="Weight decay for training")
    parser.add_argument("--report_to", type=str, default="wandb")
    parser.add_argument("--project_name", type=str, default="sft_llama3_project", help="WandB project name")
    parser.add_argument("--run_name", type=str, default=None, help="WandB run name")

    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu",
                        help="Device to use for training")
    parser.add_argument("--plot_dist", type=bool, default=False,
                        help="Whether to plot the distribution of the dataset")
    parser.add_argument("--dir_plot", type=str, default="./", 
                        help="Directory to save the plot")

    return parser.parse_args()


def prepare_sft_prompt_and_answer(row, prompt_template, tokenizer):
    
    messages = []

    system_prompt, few_shot_prompt = prompt_template.split("# Examples of using ImagePatch\n")
    system_prompt_full = (
        "You are an AI that uses a special ImagePatch class to answer questions about images.\n"
        "Here is the class definition:\n\n"
        f"{system_prompt}\n\n"
        "Please use this class to answer queries about images.\n"
        "When writing the final solution, you typically define a function:\n\n"
        "def execute_command(image)->str:\n"
        "    # put your logic here\n"
        "Your job is to produce the correct code in that function "
        "so that it answers the question or does the operation asked by the user.\n"
    )

    messages.append({"role": "system", "content": system_prompt_full})
    few_shot_prompt = few_shot_prompt.split("\n\n")[:-1]
    for example in few_shot_prompt:
        lines = example.splitlines()
        messages.append({"role": "user", "content": "\n".join(lines[:2])})
        messages.append({"role": "assistant", "content": "\n".join(lines[2:])})

    messages.append({"role": "user", "content": f"{row["prompt"]}\ndef execute_command(image)->str:"})
    
    messages.append({"role": "assistant", "content": row["output"]})

    #Verify that the messages are correct
    logger.info(f"Prompt: {messages[0]['content']}")

    return tokenizer.apply_chat_template(messages, tokenize=False)
    

def count_tokens(text, tokenizer):
    """
    Count the number of tokens in a given text using the specified tokenizer.
    """
    if isinstance(text, str):
        text = [text]
    return tokenizer(text, add_special_tokens=True, return_attention_mask=False)["input_ids"].size(1)
    

def tokenize_for_sft(examples, tokenizer, max_seq_length=8192):
    """
    Tokenize the prompt + answer together for standard supervised fine-tuning.
    Here we simply concatenate [prompt, answer] into a single sequence.

    If you want to mask out the prompt portion from the loss, you can set label IDs = -100
    for the prompt tokens. Shown below is a simpler approach that uses full cross-entropy
    on both question and answer. Adapt if desired.
    """

    # Merge prompt + answer
    merged_texts = [p + a for p, a in zip(examples["prompt"], examples["answer"])]
    tokenized = tokenizer(
        merged_texts,
        padding="longest",
        truncation=True,
        max_length=max_seq_length,
        return_tensors="pt"
    )

    # 'input_ids' and 'attention_mask' come out as lists in a batched map, so keep them separate
    return tokenized


def evaluate_dpo_loss(model, dpo_dataset, tokenizer, device="cuda"):
    """
    If you want to see how SFT influences the DPO objective on a DPO dev set,
    you can compute the DPO loss similarly to the DPO script.
    """
    logger.info("Evaluating DPO loss on DPO-style dev dataset...")

    model.eval()
    total_loss = 0.0
    total_samples = 0

    # We'll iterate the DPO dev dataset in small batches
    for sample in dpo_dataset:
        prompt = sample["prompt"]
        chosen = sample["chosen"]
        rejected = sample["rejected"]

        # Tokenize
        prompt_chosen = tokenizer(prompt + chosen, return_tensors="pt", truncation=True).to(device)
        prompt_rejected = tokenizer(prompt + rejected, return_tensors="pt", truncation=True).to(device)

        with torch.no_grad():
            # Get log probabilities
            chosen_outputs = model(**prompt_chosen)
            rejected_outputs = model(**prompt_rejected)

            # For a causal LM, the loss is the mean cross-entropy.
            # Precisely replicating the DPO loss might require more advanced logic,
            # but here's a simplified approach to measure log-likelihood difference.
            chosen_loss = chosen_outputs.loss
            rejected_loss = rejected_outputs.loss

        # Approx DPO "loss" measurement. You can adapt to replicate your exact formula.
        total_loss += (chosen_loss.item() - rejected_loss.item())
        total_samples += 1

    dpo_loss = total_loss / max(1, total_samples)
    return {"dpo_loss": dpo_loss}


def main():
    args = parse_args()
    wandb.init(project=args.project_name, name=args.run_name)

    output_dir = os.path.join(args.output_dir, datetime.datetime.now().strftime("%m-%d_%H-%M-%S"))
    logger.info(f"Results will be saved to: {output_dir}")

    logger.info("Loading model and tokenizer...")
    max_seq_length = 8192
    dtype = torch.bfloat16 if is_bfloat16_supported() else torch.float16

    # Load base model
    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=args.model_name,
        max_seq_length=max_seq_length,
        dtype=dtype,
    )

    # Convert to a PEFT LoRA model (similar to your DPO code).
    model = FastLanguageModel.get_peft_model(
        model,
        r=64,
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
        lora_alpha=64,
        lora_dropout=0,
        bias="none",
        use_gradient_checkpointing="unsloth",
        random_state=3407,
        use_rslora=False,
        loftq_config=None,
    )

    # Read the prompt template once
    with open("prompts/benchmarks/gqa.prompt", "r") as f:
        prompt_template = f.read()

    logger.info("Loading SFT train and dev datasets...")
    train_sft = datasets.load_from_disk(args.train_dataset_sft)
    dev_sft = datasets.load_from_disk(args.dev_dataset_sft)

    #Convert to pandas DataFrame

    train_sft = train_sft.to_pandas()
    dev_sft = dev_sft.to_pandas()

    train_sft.head(5)
    dev_sft.head(5)

    # Create the text column for SFT

    train_sft["text"] = train_sft.apply(prepare_sft_prompt_and_answer, axis=1, args=(prompt_template, tokenizer))
    dev_sft["text"] = dev_sft.apply(prepare_sft_prompt_and_answer, axis=1, args=(prompt_template, tokenizer))

    # Count tokens in the text column
    train_sft["num_tokens"] = train_sft["text"].apply(lambda x: count_tokens(x, tokenizer))
    dev_sft["num_tokens"] = dev_sft["text"].apply(lambda x: count_tokens(x, tokenizer))

    if args.plot_dist:
        plt.hist(train_sft.num_tokens, weights=np.ones(len(train_sft.num_tokens)) / len(train_sft.num_tokens))
        plt.gca().yaxis.set_major_formatter(PercentFormatter(1))
        plt.xlabel("Tokens")
        plt.ylabel("Percentage")
        plt.title("Token Distribution in SFT Dataset")
        plt.savefig(os.path.join(args.dir_plot, "token_distribution_train.png"))
        plt.show()

    # Optionally, you can also plot the distribution of the dev dataset
    if args.plot_dist:
        plt.hist(dev_sft.num_tokens, weights=np.ones(len(dev_sft.num_tokens)) / len(dev_sft.num_tokens))
        plt.gca().yaxis.set_major_formatter(PercentFormatter(1))
        plt.xlabel("Tokens")
        plt.ylabel("Percentage")
        plt.title("Token Distribution in SFT Dataset")
        plt.savefig(os.path.join(args.dir_plot, "token_distribution_dev.png"))
        plt.show()

    # Create the dataset for SFT
    train_sft = Dataset.from_pandas(train_sft)
    dev_sft = Dataset.from_pandas(dev_sft)

    response_template = "<|end_header_id|>"
    collator = DataCollatorForCompletionOnlyLM(
        response_template, tokenizer=tokenizer)

    

    # Standard huggingface TrainingArguments
    training_args = TrainingArguments(
        per_device_train_batch_size=args.batch_size,
        gradient_accumulation_steps=args.gradient_accumulation,
        fp16=(not is_bfloat16_supported()),
        bf16=is_bfloat16_supported(),
        logging_steps=args.logging_steps,
        evaluation_strategy="steps",
        eval_steps=args.eval_steps,
        save_steps=args.save_steps,
        max_steps=args.max_steps if args.max_steps > 0 else None,
        num_train_epochs=args.epochs if args.max_steps <= 0 else 100,  # or any large number if max_steps is controlling
        optim="adamw_torch",
        adam_beta1=0.9,
        adam_beta2=0.999,
        adam_epsilon=1e-8,
        weight_decay=args.weight_decay,
        learning_rate=args.learning_rate,
        warmup_ratio=args.warmup_ratio,
        lr_scheduler_type=args.lr_scheduler_type,
        output_dir=output_dir,
        report_to=args.report_to,
        run_name=args.run_name,
        save_total_limit=1,
        load_best_model_at_end=True,
        metric_for_best_model="eval_loss",
        greater_is_better=False
    )

    # Define a simple metric function to track cross-entropy loss and/or perplexity
    def compute_metrics(eval_pred):
        """
        eval_pred is (predictions, labels), but for causal LM training, 
        huggingface sets it to None. We'll rely on the Trainer’s built-in perplexity.
        If you want your own custom metrics, parse them here.
        """
        return {}

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_sft,
        eval_dataset=dev_sft,
        tokenizer=tokenizer,
        data_collator=data_collator,
        compute_metrics=compute_metrics
    )

    logger.info("Performing an initial evaluation on the dev_sft dataset...")
    eval_results = trainer.evaluate()
    logger.info(f"Initial SFT dev set evaluation results: {eval_results}")

    logger.info("Starting SFT training...")
    trainer.train()
    logger.info("SFT training completed.")

    logger.info("Evaluating final performance on dev_sft dataset...")
    final_eval_results = trainer.evaluate()
    logger.info(f"Final SFT dev set results: {final_eval_results}")

    # Optionally evaluate DPO loss on a separate dev dataset (if provided).
    if args.dev_dataset_dpo is not None:
        logger.info("Loading DPO dev dataset for additional eval...")
        dpo_dev = datasets.load_from_disk(args.dev_dataset_dpo)
        # Possibly reuse the "return_prompt_and_responses" logic from your existing code
        # if your DPO dev dataset has "prompt", "chosen", "rejected"
        # For example:
        #   dpo_dev = dpo_dev.map(
        #       return_prompt_and_responses,
        #       batched=True
        #   )
        dpo_loss_results = evaluate_dpo_loss(model, dpo_dev, tokenizer, args.device)
        logger.info(f"DPO-style dev set results: {dpo_loss_results}")
        # You can log them to wandb:
        wandb.log(dpo_loss_results)

    logger.info(f"Saving final model to {output_dir} ...")
    trainer.save_model(output_dir)

    logger.info("Training completed and model saved!")
    wandb.finish()


if __name__ == "__main__":
    main()
