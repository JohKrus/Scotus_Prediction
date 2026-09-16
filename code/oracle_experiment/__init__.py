"""Oracle Contamination Experiment — controlled fine-tuning to measure TDC effects.

Deliberately contaminates an open-source model (via LoRA SFT) with SCOTUS case
outcomes at varying doses and contamination types, then evaluates on three testsets:
  - Original: exact benchmark cases (measures maximum artificial gain)
  - Novel: paraphrased versions (measures soft contamination retention)
  - New: truly unseen cases from Term 25 (measures genuine generalization)

Three contamination types:
  - Exact: case + outcome + vote split
  - Soft: paraphrased case + outcome
  - Answer-augmented: case + outcome + chain-of-thought reasoning
"""
