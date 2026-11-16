# -*- coding: utf-8 -*-
"""
Synthetic + Rewritten: rewrite the 180 AML expert seed training answers at the length and detail of the synthetic
training answers, WITHOUT changing the decision, typology or facts. The rewritten answers are then mixed with the
synthetic data (aml_mixexp_sft_train.jsonl, build_mixexp_sft.py) and the models are retrained; if their FCR returns to
the synthetic-only level, the AML drop of Synthetic + Seeds is a writing-style effect.

Input : aml_expert_sft_train.jsonl (from build_expert_sft.py; system/user/assistant records, user = case + instruction)
Style : synthetic_train_sample.jsonl (2 REPORT records) and synthetic_train_sample_proceed.jsonl (2 PROCEED records),
        real records from the paper's aml90_sft_train.jsonl, shown to the rewriter as examples of the target style.
Output: aml_expert_sft_train_REWRITTEN.jsonl (same system/user, new assistant), rewrite_side_by_side.csv (for review),
        rewrite_stats.txt
Model : Claude Sonnet 5 through the OpenAI-compatible endpoint, as the graders use.  Env: SB_API_KEY, SB_WORKERS (6)
Checks: same DECISION; same TYPOLOGY line (PROCEED -> NONE); five lines present; 700-1,800 characters; no copy of
        the seed answers' template phrase; up to 3 attempts, otherwise the original answer is kept and flagged.
"""
import os, re, sys, json, csv, time, threading, statistics as st
from concurrent.futures import ThreadPoolExecutor, as_completed
try:
    import truststore; truststore.inject_into_ssl()
except Exception:
    pass
HERE = os.path.dirname(os.path.abspath(__file__))
SRC = os.environ.get("SRC", os.path.join(HERE, "aml_expert_sft_train.jsonl"))
OUT = os.path.join(HERE, "aml_expert_sft_train_REWRITTEN.jsonl")
SIDE = os.path.join(HERE, "rewrite_side_by_side.csv")

def load(p): return [json.loads(l) for l in open(p, encoding="utf-8") if l.strip()]
EX_REPORT = load(os.path.join(HERE, "synthetic_train_sample.jsonl"))
EX_PROCEED = load(os.path.join(HERE, "synthetic_train_sample_proceed.jsonl"))

SYSTEM = ("You rewrite anti-money-laundering training answers. You are given a case, an expert's correct but terse answer, "
          "and two example answers written in the fuller house style. Rewrite the expert's answer in the house style. "
          "Hard rules: keep the DECISION and TYPOLOGY lines exactly; keep every fact, rule citation and conclusion of the "
          "expert's answer; you may add only facts that appear in the case text itself; do not invent names, amounts, "
          "dates, documents or rules; do not change the decision. Match the examples' length and level of explicitness "
          "(each of MECHANISM, LOOPHOLE and DISTINGUISHING in one to three full sentences, DISTINGUISHING naming the specific "
          "facts from the case that a genuine customer would show and whether they are present here). Output ONLY the five "
          "lines DECISION / TYPOLOGY / MECHANISM / LOOPHOLE / DISTINGUISHING, nothing before or after.")

def case_text(user):   # the user turn is: case text + fixed instruction block
    i = user.find("\n\nDecide whether this should be REPORTED")
    return user[:i] if i > 0 else user

def build_prompt(rec):
    tgt = rec["messages"][2]["content"]; dec = "REPORT" if tgt.startswith("DECISION: REPORT") else "PROCEED"
    ex = EX_REPORT if dec == "REPORT" else EX_PROCEED
    exs = "\n\n".join(f"EXAMPLE {k+1} (house style)\nCASE:\n{case_text(e['messages'][1]['content'])}\n\nANSWER:\n{e['messages'][2]['content']}" for k, e in enumerate(ex[:2]))
    return (f"{exs}\n\n==========\nNOW REWRITE THIS ONE.\nCASE:\n{case_text(rec['messages'][1]['content'])}\n\n"
            f"EXPERT'S ANSWER (terse, correct):\n{tgt}\n\nRewrite the expert's answer in the house style, following the hard rules.")

LINES = ("DECISION:", "TYPOLOGY:", "MECHANISM:", "LOOPHOLE:", "DISTINGUISHING:")
def line(t, k):
    m = re.search(re.escape(k) + r"\s*(.*)", t); return (m.group(1).strip() if m else "")
def check(orig, new):
    new = new.strip()
    if not all(k in new for k in LINES): return "missing line"
    if not new.startswith("DECISION:"): return "does not start with DECISION"
    if line(new, "DECISION:").upper() != line(orig, "DECISION:").upper(): return "decision changed"
    if line(new, "TYPOLOGY:").strip().lower() != line(orig, "TYPOLOGY:").strip().lower(): return "typology changed"
    if not (700 <= len(new) <= 1800): return f"length {len(new)}"
    if re.search(r"here that is absent|here that IS present", new, re.I): return "template phrase kept"
    return ""

ERR_SHOWN = [0]
def ask(client, model, user):
    for att in range(3):
        try:
            r = client.chat.completions.create(model=model, messages=[{"role": "system", "content": SYSTEM}, {"role": "user", "content": user}], max_tokens=1500)
            return (r.choices[0].message.content or "").strip()
        except Exception as e:
            if ERR_SHOWN[0] < 3: ERR_SHOWN[0] += 1; print(f"\n  [api error] {str(e)[:300]}", flush=True)
            time.sleep(3)
    return ""

def main():
    key = os.environ.get("SB_API_KEY", "")
    if not key: sys.exit("Set SB_API_KEY (Anthropic key).")
    from openai import OpenAI
    client = OpenAI(base_url=os.environ.get("SB_JUDGE_BASE", "https://api.anthropic.com/v1/"), api_key=key, timeout=120.0, max_retries=3)
    model = os.environ.get("SB_MODEL", "claude-sonnet-5")
    recs = load(SRC); print(f"{len(recs)} expert records to rewrite with {model}")
    smoke = "--smoke" in sys.argv
    if smoke: recs = [r for r in recs if r["messages"][2]["content"].startswith("DECISION: REPORT")][:2] + [r for r in recs if r["messages"][2]["content"].startswith("DECISION: PROCEED")][:1]
    results = {}; lock = threading.Lock()
    def work(i, rec):
        orig = rec["messages"][2]["content"]; reason = ""; new = ""
        for att in range(3):
            new = ask(client, model, build_prompt(rec)); reason = check(orig, new)
            if not reason: break
        return i, new, reason
    with ThreadPoolExecutor(max_workers=int(os.environ.get("SB_WORKERS", "6"))) as pool:
        futs = [pool.submit(work, i, r) for i, r in enumerate(recs)]
        for n, fu in enumerate(as_completed(futs), 1):
            i, new, reason = fu.result()
            with lock: results[i] = (new, reason)
            print(f"\r  {n}/{len(recs)}", end="", flush=True)
    print()
    out, side, flagged = [], [], 0
    for i, rec in enumerate(recs):
        new, reason = results[i]; orig = rec["messages"][2]["content"]
        use = new if not reason else orig
        if reason: flagged += 1
        if smoke and reason: print(f"\n[raw reply for flagged record {i}] {new[:600]!r}")
        r2 = {"messages": [rec["messages"][0], rec["messages"][1], {"role": "assistant", "content": use}]}
        out.append(r2); side.append({"idx": i, "decision": line(orig, "DECISION:"), "typology": line(orig, "TYPOLOGY:"), "orig_chars": len(orig), "new_chars": len(use), "flag": reason, "original": orig, "rewritten": use})
    if smoke:
        for s in side: print(f"\n===== {s['decision']} | {s['typology']} | {s['orig_chars']} -> {s['new_chars']} chars | flag: {s['flag'] or 'ok'}\n{s['rewritten']}")
        return
    with open(OUT, "w", encoding="utf-8") as f:
        for r in out: f.write(json.dumps(r, ensure_ascii=False) + "\n")
    with open(SIDE, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(side[0].keys())); w.writeheader(); w.writerows(side)
    o = [s["orig_chars"] for s in side]; nw = [s["new_chars"] for s in side if not s["flag"]]
    stats = (f"records {len(out)} | rewritten ok {len(out)-flagged} | kept original (flagged) {flagged}\n"
             f"median chars original {int(st.median(o))} -> rewritten {int(st.median(nw)) if nw else 0}\n"
             f"template phrase in rewritten: {sum(1 for s in side if re.search(r'here that is absent', s['rewritten'], re.I))}\n")
    open(os.path.join(HERE, "rewrite_stats.txt"), "w").write(stats); print(stats)
    print("wrote", os.path.basename(OUT), "and", os.path.basename(SIDE))

if __name__ == "__main__":
    main()
