#!/usr/bin/env python3
"""Synthetic + Rewritten training file (run in the working folder): aml_mixexp_sft_train.jsonl = aml90_sft_train.jsonl (9,694 synthetic records, paper)
+ aml_expert_sft_train_REWRITTEN.jsonl (180 expert records rewritten into the synthetic style). Shuffled once, seed 42,
exactly like build_mix_sft.py did for the as-is expert records."""
import json, random, os
syn, exp, out = "aml90_sft_train.jsonl", "aml_expert_sft_train_REWRITTEN.jsonl", "aml_mixexp_sft_train.jsonl"
for f in (syn, exp):
    if not os.path.exists(f): raise SystemExit(f"missing {f}")
s = [l for l in open(syn, encoding="utf-8") if l.strip()]; e = [l for l in open(exp, encoding="utf-8") if l.strip()]
for l in s[:3] + e[:3]: json.loads(l)
allrec = s + e; random.Random(42).shuffle(allrec); open(out, "w", encoding="utf-8").writelines(allrec)
print(f"synthetic={len(s)} rewritten expert={len(e)} -> {out} = {len(allrec)} records")
