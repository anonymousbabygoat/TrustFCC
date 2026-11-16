#!/usr/bin/env python3
"""Build test_items_aml.json / test_items_market.json (the case files read by eval_synth_test.py and
run_frontier_synth_test.py) from the larger synthetic test set released with the dataset
(aml_test.csv.gz: 889 pairs; market_test.csv.gz: 721 pairs). Only the case text is kept; the reference
explanations stay in the csv and are read by grade_synth_test.py.
Usage: python3 make_test_items.py   (run in the folder holding the two csv.gz files)"""
import csv, gzip, json, os, sys
csv.field_size_limit(10**9)
for dom in ("aml", "market"):
    src = next((p for p in (f"{dom}_test.csv.gz", f"{dom}_test.csv") if os.path.exists(p)), None)
    if not src: print("missing", f"{dom}_test.csv.gz"); continue
    f = gzip.open(src, "rt", encoding="utf-8", newline="") if src.endswith(".gz") else open(src, encoding="utf-8", newline="")
    items = [{"mid": r["mid"], "seed_id": r["seed_id"], "family": r["family"], "typology": r["typology"],
              "vstory": r["violation_story"], "tstory": r["twin_story"]} for r in csv.DictReader(f)]
    json.dump(items, open(f"test_items_{dom}.json", "w", encoding="utf-8"), ensure_ascii=False)
    print(f"{dom}: {len(items)} pairs -> test_items_{dom}.json")
