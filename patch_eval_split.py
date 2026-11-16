#!/usr/bin/env python3
"""Add an optional EVAL_SPLIT filter to eval_aml90_2way.py and eval_market_2way.py (run once, in the working folder).
EVAL_SPLIT=unseen  -> answer only the held-out cases (60 AML / 50 market) instead of all 240 / 200.
Unset              -> unchanged behaviour.  Idempotent; keeps a .bak of each file."""
import os, shutil
LINE = "ITEMS=[i for i in ITEMS if not os.environ.get('EVAL_SPLIT') or i['split']==os.environ['EVAL_SPLIT']]  # EVAL_SPLIT filter\n"
for f in ("eval_aml90_2way.py", "eval_market_2way.py"):
    if not os.path.exists(f): print("missing", f); continue
    lines = open(f, encoding="utf-8").readlines()
    if any("EVAL_SPLIT" in l for l in lines): print(f, "already patched"); continue
    idx = next(i for i, l in enumerate(lines) if l.startswith("print(") and "eval:" in l)
    shutil.copy(f, f + ".bak")
    lines.insert(idx, LINE)
    open(f, "w", encoding="utf-8").writelines(lines)
    print(f, "patched (filter inserted before line", idx + 1, "); backup ->", f + ".bak")
