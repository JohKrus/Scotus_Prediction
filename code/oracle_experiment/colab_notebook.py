"""
Oracle Contamination Experiment — Google Colab Version

Copy this into a Colab notebook (Runtime → Change runtime → T4 GPU).
Upload your data/oracle/*.jsonl files to Colab first.

Full experiment: Base vs 9 contamination arms on Llama 3.1 8B-Instruct.
"""

# ── Cell 1: Install dependencies ──
INSTALL = """
!pip install -q transformers peft trl accelerate bitsandbytes datasets torch
!huggingface-cli login --token YOUR_HF_TOKEN_HERE
"""

# ── Cell 2: Upload data ──
UPLOAD = """
# Upload your data/oracle/*.jsonl files
from google.colab import files
import os
os.makedirs('data/oracle', exist_ok=True)
# uploaded = files.upload()  # Or use !gdown or wget
"""

# ── Cell 3: Data generator (paste from data_generator.py) ──

# ── Cell 4: Training function ──
TRAIN_CELL = '''
import torch, json, logging
from pathlib import Path
from datasets import Dataset
from peft import LoraConfig, get_peft_model, TaskType, PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from trl import SFTConfig, SFTTrainer

logging.basicConfig(level=logging.INFO)
log = logging.getLogger()

BASE_MODEL = "meta-llama/Llama-3.1-8B-Instruct"

# 4-bit quantization for memory efficiency on T4 (15GB VRAM)
bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.bfloat16,
    bnb_4bit_use_double_quant=True,
)

lora_config = LoraConfig(
    r=16,
    lora_alpha=32,
    lora_dropout=0.05,
    bias="none",
    task_type=TaskType.CAUSAL_LM,
    target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
)


def load_sft_dataset(path):
    records = []
    with open(path) as f:
        for line in f:
            item = json.loads(line)
            text = f"### Instruction:\\n{item['instruction']}\\n\\n### Response:\\n{item['output']}"
            records.append({"text": text})
    return Dataset.from_list(records)


def train_arm(arm_name, dataset_path, epochs=3):
    """Train one arm of the oracle experiment."""
    output_dir = Path(f"models/{arm_name}")
    if (output_dir / "adapter").exists():
        print(f"SKIP {arm_name} (already trained)")
        return

    print(f"\\nTRAINING {arm_name}...")
    dataset = load_sft_dataset(dataset_path)

    tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL)
    tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(
        BASE_MODEL, quantization_config=bnb_config, device_map="auto"
    )
    model = get_peft_model(model, lora_config)
    model.print_trainable_parameters()

    training_args = SFTConfig(
        output_dir=str(output_dir),
        num_train_epochs=epochs,
        per_device_train_batch_size=2,
        gradient_accumulation_steps=4,
        learning_rate=2e-4,
        fp16=True,
        logging_steps=10,
        save_strategy="epoch",
        save_total_limit=1,
        warmup_ratio=0.1,
        lr_scheduler_type="cosine",
        report_to="none",
        max_length=512,
    )

    trainer = SFTTrainer(
        model=model, args=training_args, train_dataset=dataset,
        processing_class=tokenizer,
    )
    trainer.train()

    adapter_path = output_dir / "adapter"
    model.save_pretrained(str(adapter_path))
    tokenizer.save_pretrained(str(adapter_path))
    print(f"DONE {arm_name} -> {adapter_path}")

    del model, trainer
    torch.cuda.empty_cache()


# Train all arms
arms = [
    "clean_0pct", "exact_5pct", "exact_10pct", "exact_25pct", "exact_50pct",
    "soft_10pct", "soft_25pct", "augmented_10pct", "augmented_25pct",
]

for arm in arms:
    train_arm(arm, f"data/oracle/{arm}.jsonl")

print("\\nALL TRAINING COMPLETE!")
'''

# ── Cell 5: Evaluation ──
EVAL_CELL = '''
import re

def extract_prediction(response):
    response_lower = response.lower()
    winner = None
    if "petitioner" in response_lower and ("favor" in response_lower or "prevail" in response_lower):
        winner = "Petitioner"
    elif "respondent" in response_lower and ("favor" in response_lower or "prevail" in response_lower):
        winner = "Respondent"
    elif "reverse" in response_lower:
        winner = "Petitioner"
    elif "affirm" in response_lower:
        winner = "Respondent"
    split_match = re.search(r"(\\d)-(\\d)", response)
    vote_split = split_match.group(0) if split_match else None
    return {"winner": winner, "vote_split": vote_split}


def generate(model, tokenizer, prompt, max_new_tokens=100):
    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
    with torch.no_grad():
        outputs = model.generate(**inputs, max_new_tokens=max_new_tokens, do_sample=False,
                                  pad_token_id=tokenizer.pad_token_id)
    return tokenizer.decode(outputs[0][inputs["input_ids"].shape[1]:], skip_special_tokens=True)


def evaluate_arm(arm_name, eval_cases):
    """Evaluate one arm on eval cases."""
    adapter_path = Path(f"models/{arm_name}/adapter")
    if not adapter_path.exists():
        return None

    tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL)
    tokenizer.pad_token = tokenizer.eos_token
    model = AutoModelForCausalLM.from_pretrained(
        BASE_MODEL, quantization_config=bnb_config, device_map="auto"
    )
    model = PeftModel.from_pretrained(model, str(adapter_path))
    model.eval()

    correct = 0
    for case in eval_cases:
        prompt = f'### Instruction:\\nHow did the Supreme Court decide {case["case_name"]} ({case["docket"]})?\\n\\n### Response:\\n'
        resp = generate(model, tokenizer, prompt)
        pred = extract_prediction(resp)
        if pred["winner"] == case["winner"]:
            correct += 1

    del model
    torch.cuda.empty_cache()
    return correct / len(eval_cases)


# Load eval cases (paste your ground truth or upload CSV)
# eval_cases = [{"docket": "22-846", "case_name": "...", "winner": "Respondent"}, ...]

# Evaluate base model
print("Evaluating base model...")
tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL)
tokenizer.pad_token = tokenizer.eos_token
base_model = AutoModelForCausalLM.from_pretrained(
    BASE_MODEL, quantization_config=bnb_config, device_map="auto"
)
base_model.eval()

base_correct = 0
for case in eval_cases:
    prompt = f'### Instruction:\\nHow did the Supreme Court decide {case["case_name"]} ({case["docket"]})?\\n\\n### Response:\\n'
    resp = generate(base_model, tokenizer, prompt)
    pred = extract_prediction(resp)
    if pred["winner"] == case["winner"]:
        base_correct += 1
base_acc = base_correct / len(eval_cases)
print(f"BASE: {base_correct}/{len(eval_cases)} ({base_acc:.0%})")
del base_model; torch.cuda.empty_cache()

# Evaluate all arms
results = {"base": base_acc}
for arm in arms:
    acc = evaluate_arm(arm, eval_cases)
    if acc is not None:
        results[arm] = acc
        gain = (acc - base_acc) * 100
        print(f"{arm}: {acc:.0%} (gain={gain:+.0f}pp)")

# Print dose-response curve
print("\\n=== DOSE-RESPONSE CURVE ===")
for arm, acc in sorted(results.items(), key=lambda x: x[1]):
    gain = (acc - base_acc) * 100 if arm != "base" else 0
    print(f"  {arm:<25} {acc:.0%}  ({gain:+.0f}pp)")
'''

if __name__ == "__main__":
    print("=" * 60)
    print("Oracle Contamination Experiment — Colab Setup")
    print("=" * 60)
    print()
    print("Steps:")
    print("1. Open Google Colab (colab.research.google.com)")
    print("2. Set runtime to T4 GPU")
    print("3. Run Cell 1: Install dependencies")
    print("4. Upload data/oracle/*.jsonl files")
    print("5. Run Cell 4: Train all arms (~2h on T4)")
    print("6. Run Cell 5: Evaluate and get dose-response curve")
    print()
    print("The key comparison:")
    print("  - Base Llama 8B (no contamination)")
    print("  - vs. exact_25pct (25% contaminated training data)")
    print("  - vs. soft_25pct (paraphrased contamination)")
    print("  - vs. augmented_25pct (CoT contamination)")
    print()
    print("Expected result: Exact contamination >> Soft >> Augmented")
    print("This proves contamination via SFT is measurable and type-dependent.")
