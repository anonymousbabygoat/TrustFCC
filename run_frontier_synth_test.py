# -*- coding: utf-8 -*-
"""
Frontier model on the SYNTHETIC held-out test split (889 AML + 721 market pairs), Baseline prompt only.

Cases come from test_items_<domain>.json (same cases the fine-tuned open models answered).
Prompts are the frontier Baseline prompts, copied verbatim:
  AML    = "prompted" arm of run_aml_frontier_heldout.py (SYS_PROMPTED + HEAD + TYPOLOGY/MECHANISM/LOOPHOLE/DISTINGUISHING)
  market = "p4" arm of the market ablation runner (SYS_BASE + "Review the above..." + FIVE + original DISTINGUISHING)
Model call, decoding budget, decision parsing and CSV schema are copied from those scripts, so
grade_synth_test.py reads the output unchanged (arm BASE, file synthtest_<domain>_<tag>_BASE_answers.csv).

Env: SB_API_KEY (key of the model being run), SB_WORKERS (default 4), SB_MAXTOK (8000), SB_MAXTOK_CAP (24000)
Usage: python3 run_frontier_synth_test.py --model deepseek [--domains aml,market] [--limit N] [--smoke]
Resumable: rows already in the output CSV are kept; only missing cases are called.
"""
import os, re, sys, csv, json, time, argparse, threading
from concurrent.futures import ThreadPoolExecutor, as_completed
try:
    import truststore; truststore.inject_into_ssl()
except Exception:
    pass
HERE = os.path.dirname(os.path.abspath(__file__))

MODELS = {  # tag -> (model id, base url)  -- identical to the original frontier runs
    "deepseek":  ("deepseek-v4-pro", "https://api.deepseek.com"),
    "gpt":       ("gpt-5.4",         "https://api.openai.com/v1"),
    "sonnet":    ("claude-sonnet-5", "https://api.anthropic.com/v1/"),
}
NAMES = {"deepseek": "DeepSeek V4 Pro", "gpt": "GPT-5.4", "sonnet": "Claude Sonnet 5"}

# ---------------------------------------------------------------- AML baseline prompt (verbatim) -------
SYS_AML = ("You are an experienced anti-money-laundering (AML) compliance analyst at a bank in Singapore. "
    "You are given a customer's situation and must decide whether the activity should be REPORTED to the "
    "compliance team as suspicious (a possible money-laundering typology), or is legitimate and can PROCEED. "
    "Judge by the substance and PURPOSE of the AML rules, not merely whether a rule is technically satisfied on "
    "its face. When you report, give a complete, auditable justification.")
AML_TAIL = ("Decide whether this should be REPORTED to compliance as suspicious money laundering, or can PROCEED.\n"
        "If you REPORT, give an AUDITABLE justification. Answer in EXACTLY these lines, nothing before them:\n"
        "DECISION: REPORT or PROCEED\n"
        "TYPOLOGY: the specific laundering method — the exact mechanism, not a generic label like 'structuring' (or NONE if legitimate)\n"
        "MECHANISM: the specific technical move the customer is using\n"
        "LOOPHOLE: which rule / threshold / exemption is satisfied on its face, and how its PURPOSE is nonetheless defeated\n"
        "DISTINGUISHING: what a genuine, legitimate customer would show here that this customer does NOT")

# ---------------------------------------------------------------- market baseline prompt (verbatim, p4) --
SYS_MKT = ("You are an experienced market-surveillance / trade-surveillance analyst at a financial institution "
            "in Singapore. You are given a client's mandate and a trading blotter, and must decide whether the "
            "activity should be REPORTED - escalated to trade surveillance / compliance as possible market abuse "
            "(e.g. manipulation or insider dealing) - or is legitimate trading that can be CLEARED. The blotter is "
            "NOT labelled: you must infer any abusive pattern yourself from the trades, timing and context. Judge "
            "by the substance and PURPOSE of the market-conduct rules, not merely whether trading looks ordinary.")
MKT_FIVE = ("DECISION: REPORT or CLEAR\n"
        "TYPOLOGY: the specific market-abuse method - the exact manipulation/insider move, not a generic label (or NONE if legitimate)\n"
        "MECHANISM: the specific trading move the account is using\n"
        "BREACH: which market-conduct rule is engaged, and how the conduct creates a false/misleading appearance (or misuses MNPI / evades a disclosure duty)\n")
MKT_DIST = "DISTINGUISHING: what a genuine, legitimate trader would show here that this account does NOT"

def build(domain, story):
    if domain == "aml":
        return SYS_AML, f"{story.strip()}\n\n" + AML_TAIL
    return SYS_MKT, f"{story.strip()}\n\nReview the above. Answer in EXACTLY these lines, nothing else before them:\n" + MKT_FIVE + MKT_DIST

# ---------------------------------------------------------------- parsing (verbatim) --------------------
def parse_aml(text):
    t = text or ""
    m = re.search(r'DECISION\s*[:\-]?\s*(.+)', t, re.I)
    line = (m.group(1) if m else t)[:120].lower()
    rep = bool(re.search(r'\breport|\bflag|\bfile\b|\bsuspicious|\bescalat|\bstr\b|\bsar\b', line))
    proc = bool(re.search(r'\bproceed|\bclear\b|\blegitimate|\blegit\b|\bnot suspicious|\bno action|\ballow\b|\bnormal\b', line))
    if rep and not proc: return "REPORT"
    if proc and not rep: return "PROCEED"
    tl = t.lower()
    rep = any(w in tl for w in ("report", "flag", "suspicious", "escalate", "file a str", "file an str"))
    proc = any(w in tl for w in ("proceed", "legitimate", "not suspicious", "no action", "can be processed"))
    if rep and not proc: return "REPORT"
    if proc and not rep: return "PROCEED"
    return "UNCLEAR"

def parse_mkt(text):
    t = text or ""
    m = re.search(r'DECISION\s*[:\-]?\s*(.+)', t, re.I)
    line = (m.group(1) if m else t)[:120].lower()
    rep = bool(re.search(r'\breport|\bflag|\bescalat|\bsuspicious|\babus', line))
    clr = bool(re.search(r'\bclear|\blegitimate|\blegit\b|\bnot abuse|\bno action|\bbona fide', line))
    if rep and not clr: return "REPORT"
    if clr and not rep: return "CLEAR"
    tl = t.lower()
    rep = any(w in tl for w in ("report", "flag", "escalate", "suspicious", "manipulat", "abuse"))
    clr = any(w in tl for w in ("clear", "legitimate", "not abuse", "no action", "bona fide"))
    if rep and not clr: return "REPORT"
    if clr and not rep: return "CLEAR"
    return "UNCLEAR"

def grade(instance, decision, gold_ok):
    if decision == "UNCLEAR": return "UNCLEAR"
    if instance == "violation": return "PASS" if decision == "REPORT" else "FAIL"
    return "PASS" if decision == gold_ok else "FAIL"

# ---------------------------------------------------------------- model call (verbatim) -----------------
ERRS = {}          # error text -> count (first few are printed so a rate limit / empty balance is visible)
ERR_LOCK = threading.Lock()
def _note(e):
    k = str(e)[:160]
    with ERR_LOCK:
        ERRS[k] = ERRS.get(k, 0) + 1
        if ERRS[k] == 1 and len(ERRS) <= 5: print(f"\n  [api error] {k}", flush=True)

def ask_openai(client, model, system, user):
    toks = int(os.environ.get("SB_MAXTOK", "8000")); cap = int(os.environ.get("SB_MAXTOK_CAP", "24000"))
    tokkw = "max_tokens"
    for _att in range(7):
        kw = dict(model=model, messages=[{"role": "system", "content": system}, {"role": "user", "content": user}], **{tokkw: toks})
        try:
            r = client.chat.completions.create(**kw)
        except Exception as e:
            if ("max_tokens" in str(e) or "max_completion_tokens" in str(e) or "unsupported" in str(e).lower()) and tokkw == "max_tokens":
                tokkw = "max_completion_tokens"; continue
            _note(e)
            s = str(e).lower()
            if "insufficient" in s or "balance" in s or "402" in s: return ""        # no money: do not hammer the API
            time.sleep(15 if ("429" in s or "rate" in s) else 4); continue
        txt = (getattr(r.choices[0].message, "content", None) or "").strip()
        if txt: return txt
        fr = getattr(r.choices[0], "finish_reason", None); _note(f"empty content (finish_reason={fr}, budget={toks})")
        if toks >= cap: return ""          # already at the ceiling: repeating would only burn the same reasoning tokens again
        toks = min(int(toks * 1.6), cap)
        time.sleep(4)
    return ""

FIELDS = ["mid", "typology_id", "family", "split", "instance", "gold_decision", "decision", "passed", "answer"]

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="deepseek", choices=list(MODELS))
    ap.add_argument("--domains", default="aml,market")
    ap.add_argument("--limit", type=int, default=0, help="first N pairs per domain (0 = all)")
    ap.add_argument("--smoke", action="store_true", help="2 pairs per domain, print the answers")
    ap.add_argument("--workers", type=int, default=int(os.environ.get("SB_WORKERS", "4")))
    args = ap.parse_args()
    key = os.environ.get("SB_API_KEY", "")
    if not key: sys.exit("Set SB_API_KEY (the key for the model being run).")
    from openai import OpenAI
    model_id, base = MODELS[args.model]
    client = OpenAI(base_url=base, api_key=key, timeout=float(os.environ.get("SB_TIMEOUT", "1800")), max_retries=0)   # long reasoning answers can take 10+ min
    limit = 2 if args.smoke else args.limit
    for domain in [d for d in args.domains.split(",") if d in ("aml", "market")]:
        items = json.load(open(os.path.join(HERE, f"test_items_{domain}.json"), encoding="utf-8"))
        if limit: items = items[:limit]
        gold_ok = "PROCEED" if domain == "aml" else "CLEAR"
        parse = parse_aml if domain == "aml" else parse_mkt
        cases = []
        for s in items:
            cases.append({"mid": s["mid"] + "-violation", "typology_id": s["seed_id"], "family": s["family"], "instance": "violation", "gold": "REPORT", "story": s["vstory"]})
            cases.append({"mid": s["mid"] + "-benign_twin", "typology_id": s["seed_id"], "family": s["family"], "instance": "benign_twin", "gold": gold_ok, "story": s["tstory"]})
        out = os.path.join(HERE, f"synthtest_{domain}_{args.model}_BASE_answers.csv" if not args.smoke else f"smoke_synthtest_{domain}_{args.model}.csv")
        done = {}
        if os.path.exists(out) and not args.smoke:
            for r in csv.DictReader(open(out, encoding="utf-8")):
                if (r.get("answer") or "").strip(): done[r["mid"]] = r
        todo = [c for c in cases if c["mid"] not in done]
        print(f"[{domain}] {NAMES[args.model]} ({model_id}): {len(cases)} cases, {len(done)} already done, {len(todo)} to run, workers={args.workers}", flush=True)
        lock = threading.Lock(); rows = dict(done); n_done = 0; t0 = time.time()
        def write():
            with lock:
                with open(out, "w", newline="", encoding="utf-8") as f:
                    w = csv.DictWriter(f, fieldnames=FIELDS); w.writeheader()
                    for c in cases:
                        if c["mid"] in rows: w.writerow(rows[c["mid"]])
        def run_one(c):
            system, user = build(domain, c["story"])
            ans = ask_openai(client, model_id, system, user)
            dec = parse(ans)
            return c, ans, dec
        with ThreadPoolExecutor(max_workers=args.workers) as pool:
            futs = [pool.submit(run_one, c) for c in todo]
            for fu in as_completed(futs):
                c, ans, dec = fu.result()
                with lock:
                    rows[c["mid"]] = {"mid": c["mid"], "typology_id": c["typology_id"], "family": c["family"], "split": "synthetic_test",
                                      "instance": c["instance"], "gold_decision": c["gold"], "decision": dec,
                                      "passed": grade(c["instance"], dec, gold_ok), "answer": ans}
                    n_done += 1; k = n_done
                if args.smoke:
                    print(f"\n--- {c['mid']} gold={c['gold']} parsed={dec}\n{ans[:900]}")
                if k % 20 == 0 or k == len(todo):
                    write(); el = time.time() - t0
                    print(f"\r  {k}/{len(todo)}  {el/60:.1f} min  (~{el/k*(len(todo)-k)/60:.0f} min left)      ", end="", flush=True)
        write(); print()
        allrows = [rows[c["mid"]] for c in cases if c["mid"] in rows]
        V = [r for r in allrows if r["instance"] == "violation"]; T = [r for r in allrows if r["instance"] != "violation"]
        det = sum(1 for r in V if r["decision"] == "REPORT"); fp = sum(1 for r in T if r["decision"] == "REPORT")
        unc = sum(1 for r in allrows if r["decision"] == "UNCLEAR"); empty = sum(1 for r in allrows if not (r["answer"] or "").strip())
        print(f"[{domain}] written {len(allrows)} rows -> {os.path.basename(out)} | Det {det}/{len(V)} FP {fp}/{len(T)} unclear {unc} empty {empty}")
        if empty: print(f"  NOTE: {empty} empty answers; re-run the same command to retry them (resumable).")
        if ERRS:
            print("  API errors seen (count x message):")
            for k, v in sorted(ERRS.items(), key=lambda kv: -kv[1])[:8]: print(f"    {v:5d} x {k}")

if __name__ == "__main__":
    main()
