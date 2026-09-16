"""LoRA fine-tuning pipeline for oracle contamination experiment.

Uses QLoRA on Apple Silicon (MPS) or CUDA to fine-tune an open-source model
with controlled contamination data.

Usage:
    python -m oracle_experiment.fine_tune \
        --base-model meta-llama/Llama-3.1-8B-Instruct \
        --dataset data/oracle/exact_10pct.jsonl \
        --output models/oracle_exact_10pct \
        --epochs 3
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path

import torch
from datasets import Dataset
from peft import LoraConfig, get_peft_model, TaskType
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
)
from trl import SFTConfig
from trl import SFTTrainer

log = logging.getLogger(__name__)

# Default LoRA config optimized for legal domain fine-tuning
DEFAULT_LORA_CONFIG = LoraConfig(
    r=16,
    lora_alpha=32,
    lora_dropout=0.05,
    bias="none",
    task_type=TaskType.CAUSAL_LM,
    target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
)


def load_sft_dataset(path: Path) -> Dataset:
    """Load JSONL dataset and convert to HuggingFace Dataset."""
    records = []
    with open(path) as f:
        for line in f:
            item = json.loads(line)
            text = f"### Instruction:\n{item['instruction']}\n\n### Response:\n{item['output']}"
            records.append({"text": text})
    return Dataset.from_list(records)


def get_device():
    """Get best available device."""
    if torch.cuda.is_available():
        return "cuda"
    elif torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def fine_tune(
    base_model: str,
    dataset_path: Path,
    output_dir: Path,
    epochs: int = 3,
    batch_size: int = 2,
    gradient_accumulation: int = 8,
    learning_rate: float = 2e-4,
    max_seq_length: int = 1024,
    lora_config: LoraConfig | None = None,
) -> Path:
    """Run LoRA fine-tuning.

    Args:
        base_model: HuggingFace model ID (e.g., "meta-llama/Llama-3.1-8B-Instruct")
        dataset_path: Path to JSONL training data
        output_dir: Where to save the LoRA adapter
        epochs: Number of training epochs
        batch_size: Per-device batch size
        gradient_accumulation: Gradient accumulation steps (effective batch = batch_size * grad_accum)
        learning_rate: Learning rate
        max_seq_length: Maximum sequence length
        lora_config: Custom LoRA configuration

    Returns:
        Path to the saved adapter
    """
    device = get_device()
    log.info(f"Device: {device}")
    log.info(f"Base model: {base_model}")
    log.info(f"Dataset: {dataset_path}")

    # Load dataset
    dataset = load_sft_dataset(dataset_path)
    log.info(f"Dataset size: {len(dataset)} examples")

    # Load tokenizer
    tokenizer = AutoTokenizer.from_pretrained(base_model)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # Load model — use bfloat16 on MPS (float16 causes NaN gradients on Apple Silicon)
    if device == "mps":
        model_kwargs = {"torch_dtype": torch.bfloat16}
    else:
        model_kwargs = {"torch_dtype": torch.float16, "device_map": "auto"}

    model = AutoModelForCausalLM.from_pretrained(base_model, **model_kwargs)

    if device == "mps":
        model = model.to("mps")

    # Apply LoRA
    if lora_config is None:
        lora_config = DEFAULT_LORA_CONFIG

    model = get_peft_model(model, lora_config)
    model.print_trainable_parameters()

    # Training arguments
    output_dir.mkdir(parents=True, exist_ok=True)

    training_args = SFTConfig(
        output_dir=str(output_dir),
        num_train_epochs=epochs,
        per_device_train_batch_size=batch_size,
        gradient_accumulation_steps=gradient_accumulation,
        learning_rate=learning_rate,
        fp16=device == "cuda",
        bf16=device == "mps",
        logging_steps=10,
        save_strategy="epoch",
        save_total_limit=2,
        warmup_ratio=0.1,
        lr_scheduler_type="cosine",
        report_to="none",
        max_length=max_seq_length,
    )

    # SFT Trainer
    trainer = SFTTrainer(
        model=model,
        args=training_args,
        train_dataset=dataset,
        processing_class=tokenizer,
    )

    # Train
    log.info("Starting training...")
    trainer.train()

    # Save adapter
    adapter_path = output_dir / "adapter"
    model.save_pretrained(str(adapter_path))
    tokenizer.save_pretrained(str(adapter_path))
    log.info(f"Adapter saved to {adapter_path}")

    # Save training metadata
    meta = {
        "base_model": base_model,
        "dataset": str(dataset_path),
        "dataset_size": len(dataset),
        "epochs": epochs,
        "learning_rate": learning_rate,
        "lora_r": lora_config.r,
        "lora_alpha": lora_config.lora_alpha,
        "device": device,
    }
    with open(output_dir / "training_meta.json", "w") as f:
        json.dump(meta, f, indent=2)

    return adapter_path


def load_oracle_model(base_model: str, adapter_path: Path):
    """Load a fine-tuned oracle model for inference."""
    from peft import PeftModel

    device = get_device()
    tokenizer = AutoTokenizer.from_pretrained(base_model)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    dtype = torch.bfloat16 if device == "mps" else torch.float16
    model = AutoModelForCausalLM.from_pretrained(base_model, torch_dtype=dtype)
    model = PeftModel.from_pretrained(model, str(adapter_path))

    if device == "mps":
        model = model.to("mps")
    elif device == "cuda":
        model = model.to("cuda")

    model.eval()
    return model, tokenizer


def generate(model, tokenizer, prompt: str, max_new_tokens: int = 512) -> str:
    """Generate a response from the oracle model."""
    device = next(model.parameters()).device
    inputs = tokenizer(prompt, return_tensors="pt").to(device)

    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=False,
            pad_token_id=tokenizer.pad_token_id,
        )

    response = tokenizer.decode(outputs[0][inputs["input_ids"].shape[1]:], skip_special_tokens=True)
    return response.strip()


# ══════════════════════════════════════════════════════════════════════════
# CLI
# ══════════════════════════════════════════════════════════════════════════

def main():
    parser = argparse.ArgumentParser(description="Oracle Contamination Fine-Tuning")
    parser.add_argument("--base-model", type=str, default="meta-llama/Llama-3.1-8B-Instruct")
    parser.add_argument("--dataset", type=str, required=True)
    parser.add_argument("--output", type=str, required=True)
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--lr", type=float, default=2e-4)
    parser.add_argument("--max-seq-length", type=int, default=1024)

    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)-7s %(message)s")

    fine_tune(
        base_model=args.base_model,
        dataset_path=Path(args.dataset),
        output_dir=Path(args.output),
        epochs=args.epochs,
        batch_size=args.batch_size,
        learning_rate=args.lr,
        max_seq_length=args.max_seq_length,
    )


if __name__ == "__main__":
    main()
