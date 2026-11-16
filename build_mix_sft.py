#!/usr/bin/env python3
"""Build the Synthetic + Seeds training files (run once in the working folder):
   aml_mix_sft_train.jsonl    = aml90_sft_train.jsonl    (9,694 synthetic records, paper) + aml_expert_sft_train.jsonl    (180 expert records)
   market_mix_sft_train.jsonl = market75_sft_train.jsonl (9,516 synthetic records, paper) + market_expert_sft_train.jsonl (150 expert records)
Records are concatenated and shuffled once with seed 42 (the Trainer shuffles every epoch anyway). Nothing else
changes: same record format, same prompts, same assistant target. Validation = the paper's synthetic val file
(aml90_sft_val.jsonl / market75_sft_val.jsonl), so checkpoint selection is identical to the paper's SFT."""
import json, random, os
PAIRS = {"aml": ("aml90_sft_train.jsonl", "aml_expert_sft_train.jsonl", "aml_mix_sft_train.jsonl"),
         "market": ("market75_sft_train.jsonl", "market_expert_sft_train.jsonl", "market_mix_sft_train.jsonl")}
for dom, (syn, exp, out) in PAIRS.items():
    for f in (syn, exp):
        if not os.path.exists(f): raise SystemExit(f"missing {f} (run from the working folder)")
    s = [l for l in open(syn, encoding="utf-8") if l.strip()]
    e = [l for l in open(exp, encoding="utf-8") if l.strip()]
    for l in s[:3] + e[:3]: json.loads(l)   # sanity: valid JSON, same format
    allrec = s + e; random.Random(42).shuffle(allrec)
    open(out, "w", encoding="utf-8").writelines(allrec)
    print(f"[{dom}] synthetic={len(s)} expert={len(e)} -> {out} = {len(allrec)} records")
