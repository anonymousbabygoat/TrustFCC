#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Expert Seeds SFT data: fine-tune the seven open models on ONLY the lawyer-written seed
pairs of the TRAINING typologies (90 AML, 75 market), then evaluate on the same 30 / 25 held-out typologies as
the other settings. Same 75/25 split as the paper; the held-out seeds are never used here.

Mirrors build_aml90_sft.py / build_market75_sft.py exactly (same SYSTEM, same INSTR suffix, same assistant
target distilled from the rubric's numbered elements), so the only thing that changes versus the paper's SFT runs
is the training data: 180 AML records (90 pairs x violation+twin) instead of 9,694, and 150 market records
(75 pairs) instead of 9,516.

Source: seeds_aml.json / seeds_market.json (the seed pairs embedded in eval_aml90_2way.py / eval_market_2way.py,
fields id, family, typology, split, vstory, vrubric, tstory, trubric; split = seen | unseen).

Validation file: the trainer (train_aml_sft.py) needs a VAL_FILE for eval_loss / load_best_model_at_end. With
only 90 pairs we do not want to lose any training pairs, so VAL = 10 pairs (20 records) drawn from the TRAINING
records (a subset of train, not disjoint). It only picks the checkpoint; it never touches the held-out seeds.

Usage:  python3 build_expert_sft.py            (writes 4 JSONL files next to this script)
"""
import json, re, random, os
from collections import Counter

HERE = os.path.dirname(os.path.abspath(__file__))

CFG = {
  "aml": {
    "seeds": "seeds_aml.json",
    "system": ("You are an experienced AML/CFT analyst at a bank regulated in Singapore. You are given a customer's "
               "profile and transaction record and must decide whether the activity should be REPORTED to compliance as "
               "suspicious (for a possible STR), or is legitimate and can PROCEED. Judge by the substance and PURPOSE of "
               "the AML rules (MAS Notice 626; CDSA; FATF standards), not merely whether a rule is technically satisfied "
               "on its face. When you report, give a complete, auditable justification."),
    "instr": ("\n\nDecide whether this should be REPORTED to compliance as suspicious, or can PROCEED as legitimate.\n"
              "Answer in EXACTLY these lines, nothing before them:\n"
              "DECISION: REPORT or PROCEED\n"
              "TYPOLOGY: the specific laundering typology (or NONE if legitimate)\n"
              "MECHANISM: the specific technical move the customer is using\n"
              "LOOPHOLE: which rule/threshold/exemption is met on its face and how its purpose is defeated (cite MAS 626 / CDSA)\n"
              "DISTINGUISHING: the specific fact(s) that separate this from a genuine, legitimate look-alike"),
    "rule": "LOOPHOLE", "neg": "PROCEED",
    "out": "aml_expert_sft",
  },
  "market": {
    "seeds": "seeds_market.json",
    "system": ("You are an experienced market surveillance analyst at a firm regulated in Singapore. You are given a "
               "trading and order record and must decide whether the activity should be REPORTED to surveillance as "
               "market abuse (for escalation), or is legitimate and can be CLEARED. Judge by the substance and PURPOSE "
               "of the market-abuse rules (SFA Part XII; EU MAR; IOSCO), not merely whether a rule is technically "
               "satisfied on its face. When you report, give a complete, auditable justification."),
    "instr": ("\n\nDecide whether this should be REPORTED to surveillance as market abuse, or can be CLEARED as legitimate.\n"
              "Answer in EXACTLY these lines, nothing before them:\n"
              "DECISION: REPORT or CLEAR\n"
              "TYPOLOGY: the specific market-abuse typology (or NONE if legitimate)\n"
              "MECHANISM: the specific abusive move the trader is using\n"
              "BREACH: which statutory provision is breached and why (cite SFA s.197/198/201 / MAR Art 8/12/15 / IOSCO)\n"
              "DISTINGUISHING: the specific fact(s) that separate this from a genuine, legitimate look-alike"),
    "rule": "BREACH", "neg": "CLEAR",
    "out": "market_expert_sft",
  },
}

def items(rubric):
    body = rubric.split("ANSWER KEY")[0]
    body = re.sub(r'\bEND\s*$', '', body.strip())
    # seeds S1-S100 put each element on its own line; the 13-Sep additions (S101-S120) run "1. ... 2. ..." on one
    # line, so split on "<start or space>N. <Capital>" rather than on line starts.
    parts = re.split(r'(?:(?<=^)|(?<=\s))([1-4])\.\s+(?=[A-Z])', body)   # ['pre','1',t1,'2',t2,...]
    d = {}
    for i in range(1, len(parts) - 1, 2):
        d[parts[i]] = parts[i + 1].strip()
    return d

def substance(txt):
    m = re.split(r'\s[—–-]\s', txt, maxsplit=1)
    s = m[1] if len(m) > 1 else txt
    s = re.sub(r'\s+', ' ', s).strip()
    s = re.sub(r'^(Name|Identify|Explain|Cite|State|Describe|Reach|Show)\s+(the\s+|that\s+)?', '', s, flags=re.I)
    return s.strip().rstrip('.')

def viol_answer(c, typology, vrubric):
    d = items(vrubric)
    return ("DECISION: REPORT\n"
            f"TYPOLOGY: {typology}\n"
            f"MECHANISM: {substance(d.get('1', ''))}\n"
            f"{c['rule']}: {substance(d.get('2', ''))}\n"
            f"DISTINGUISHING: {substance(d.get('3', ''))}")

def twin_answer(c, trubric):
    d = items(trubric)
    dist = (substance(d.get('2', '')) + " " + substance(d.get('3', ''))).strip()
    return (f"DECISION: {c['neg']}\n"
            "TYPOLOGY: NONE\n"
            f"MECHANISM: {substance(d.get('1', ''))}\n"
            f"{c['rule']}: N/A\n"
            f"DISTINGUISHING: {dist}")

def rec(c, story, answer):
    return {"messages": [
        {"role": "system", "content": c["system"]},
        {"role": "user", "content": story.strip() + c["instr"]},
        {"role": "assistant", "content": answer}]}

def full(c, a):
    d = dict(re.findall(r'(?m)^([A-Z]+):\s*(.*)$', a))
    keys = ["MECHANISM", c["rule"], "DISTINGUISHING"] if a.startswith("DECISION: REPORT") else ["MECHANISM", "DISTINGUISHING"]
    return all(d.get(k, "").strip() and d.get(k, "").strip() != "N/A" for k in keys)

for domain, c in CFG.items():
    seeds = json.load(open(os.path.join(HERE, c["seeds"]), encoding="utf-8"))
    seen = [s for s in seeds if s["split"] == "seen"]
    unseen = [s for s in seeds if s["split"] == "unseen"]
    built, bad = [], []
    for s in seen:
        va, ta = viol_answer(c, s["typology"], s["vrubric"]), twin_answer(c, s["trubric"])
        if not (full(c, va) and full(c, ta)):
            bad.append(s["id"]); continue
        built.append((s["id"], rec(c, s["vstory"], va), rec(c, s["tstory"], ta)))
    assert not bad, f"{domain}: rubric parse failed for {bad}"
    random.seed(42)
    train = [r for _, rv, rt in built for r in (rv, rt)]
    random.shuffle(train)
    val_ids = set(random.sample([b[0] for b in built], 10))
    val = [r for sid, rv, rt in built if sid in val_ids for r in (rv, rt)]
    tp = os.path.join(HERE, c["out"] + "_train.jsonl"); vp = os.path.join(HERE, c["out"] + "_val.jsonl")
    with open(tp, "w", encoding="utf-8") as f:
        for r in train: f.write(json.dumps(r, ensure_ascii=False) + "\n")
    with open(vp, "w", encoding="utf-8") as f:
        for r in val: f.write(json.dumps(r, ensure_ascii=False) + "\n")
    bal = Counter(r["messages"][2]["content"].split("\n")[0] for r in train)
    print(f"[{domain}] seeds={len(seeds)} seen={len(seen)} unseen={len(unseen)} (never used) | "
          f"train records={len(train)} {dict(bal)} | val records={len(val)} (subset of train) -> {os.path.basename(tp)}, {os.path.basename(vp)}")
