# -*- coding: utf-8 -*-
"""
Shared test set: open models (fine-tuned) and frontier models (all prompt arms) scored on the SAME held-out
cases (30 AML typologies, 25 market-abuse typologies) with ONE judge pass (Claude Sonnet 5, one call per element,
whole answer shown) and ONE rubric set (the expert rubrics).

Inputs (all answer CSVs already on disk; nothing is re-generated here):
  AML  frontier : exam_v3_<arm>_<tag>_heldout30.csv   (from run_aml_frontier_heldout.py; falls back to
                  exam_v3_<arm>_<tag>.csv filtered to the held-out ids, with a warning, if the heldout30 file is absent)
  AML  open FT  : aml90_2way_<tag>_LORA_answers.csv  (split == unseen)        [+ BASE with --include-base]
  MKT  frontier : market100_<arm>_<tag>.csv           (filtered to the 25 held-out ids; stories identical)
  MKT  open FT  : market75_2way_<tag>_LORA_answers.csv (split == unseen)      [+ BASE with --include-base]

Outputs (in --outdir):
  shared_heldout_percase_grades.csv  one row per violation answer: mech / loophole|breach / distinguishing / decision / fully
  shared_heldout_summary.csv         one row per (domain, group, model, arm): Det, FP, Mec, Loo|Bre, Dis, Ful  (+ n)
  shared_heldout_table.tex           LaTeX table body in the paper's column order
  judge_cache.json                   per-answer judge verdicts (re-runs are free; delete to re-judge)

Judge prompt: the system messages and element descriptions below are the ones reproduced in the paper's appendix
(--market-labels frontier, the default, is the wording used for every market-abuse table in the paper; 'open' is an
older variant kept for reference only). Detection / FP are deterministic (decision column).

Fine-tuning arms (answer files from eval_aml90_2way.py / eval_market_2way.py):
  LORA = Synthetic, EXPERT = Expert Seeds (--include-expert), MIX = Synthetic + Seeds (--include-mix),
  MIXEXP = Synthetic + Rewritten (--include-mixexp, AML only), REPRO = repeated Synthetic run (--include-repro).

Open-model answers are judged on their first complete answer: a few models (mainly Gemma-2-9B) do not stop at the end
of their turn and append a second answer; that appended text is cut before judging (see trim_runon). Pass
--keep-runon to judge the raw text instead.
"""
import os, re, sys, csv, json, time, argparse, hashlib, threading
from concurrent.futures import ThreadPoolExecutor, as_completed
try:
    import truststore; truststore.inject_into_ssl()
except Exception:
    pass

HERE = os.path.dirname(os.path.abspath(__file__))
HELD = json.load(open(os.path.join(HERE, "heldout_ids.json")))
RUB = {"aml": json.load(open(os.path.join(HERE, "rubrics_aml.json"), encoding="utf-8")),
       "market": json.load(open(os.path.join(HERE, "rubrics_market.json"), encoding="utf-8"))}

FRONTIER = [("GPT-5.4", "gpt"), ("Claude Sonnet 5", "sonnet"), ("Gemini 3.1 Pro", "geminipro"), ("DeepSeek V4 Pro", "deepseek")]
AML_ARMS = [("Baseline", "prompted"), ("+Look-alike", "p1"), ("+Reworded", "p2"), ("Contrast", "contrast")]
MKT_ARMS = [("Baseline", "p4"), ("+Innocent", "p1"), ("+Inn+Def", "p2"), ("Reworded", "p3"), ("IHF", "ihf4")]
OPEN = [("Llama-3.1-8B", "llama8b"), ("Gemma-2-9B", "gemma9b"), ("Qwen2.5-7B", "qwen7b"), ("Qwen2.5-3B", "qwen3b"),
        ("Mistral-7B", "mistral7b"), ("Gemma-2-2B", "gemma2b"), ("Llama-3.2-3B", "llama3b")]
OPEN_ARMS_MODELS = [("Gemma-2-9B (base)", "gemma9b"), ("Llama-3.1-8B (base)", "llama8b"), ("Qwen2.5-7B (base)", "qwen7b"), ("Qwen2.5-3B (base)", "qwen3b"), ("Llama-3.2-3B (base)", "llama3b"), ("Gemma-2-2B (base)", "gemma2b"), ("Mistral-7B (base)", "mistral7b")]
OPEN_FILES = {  # the AML qwen files have no tag in their names
    "aml": {"qwen7b": "aml90_2way_{arm}_answers.csv", "qwen3b": "aml90_2way_3b_{arm}_answers.csv"},
    "market": {},
}
def open_file(domain, tag, arm):  # arm = BASE | LORA | EXPERT | MIX (EXPERT = LoRA on expert seed pairs only; MIX = synthetic + expert pairs)
    pat = OPEN_FILES[domain].get(tag)
    if pat and arm not in ("EXPERT", "MIX", "REPRO", "MIXEXP"): return pat.format(arm=arm)   # EXPERT/MIX files are always tagged, qwen included
    return (f"aml90_2way_{tag}_{arm}_answers.csv" if domain == "aml" else f"market75_2way_{tag}_{arm}_answers.csv")

# ---------------------------------------------------------------- judge (verbatim from the graders) -----
JSYS = {
 "aml": ("You are a strict Singapore AML examiner. You are shown the correct rubric and an analyst's full answer, and "
         "asked about ONE element only. Judge only that element; ignore whether the other elements are present. A vague "
         "or generic statement does not earn it."),
 "market_frontier": ("You are a strict Singapore market-surveillance examiner. You are shown the correct rubric and an analyst's "
         "full answer, and asked about ONE element only. Judge only that element; ignore the others. A vague or "
         "generic statement does not earn it."),
 "market_open": ("You are a strict Singapore market-abuse examiner. You are shown the correct rubric and an analyst's full answer, "
         "and asked about ONE element only. Judge only that element; ignore whether the other elements are present. A vague "
         "or generic statement does not earn it."),
}
LABELS = {
 "aml": {
  "mech": "MECHANISM — the specific laundering method / technical move (not a generic label like 'structuring')",
  "loophole": "LOOPHOLE — a determinate rule/threshold/exemption/definition/timing/document satisfied on its face, but its PURPOSE defeated",
  "distinguishing": "DISTINGUISHING — what a genuine legitimate customer would show here that this customer does NOT (the contrast with the innocent look-alike)"},
 "market_frontier": {
  "mech": "MECHANISM - the specific market-abuse method (e.g. spoofing, layering, wash trading, marking the close), not a generic label like 'manipulation'",
  "breach": "BREACH - why the conduct breaches market-conduct rules: it creates a FALSE or MISLEADING appearance of the market, OR misuses material non-public information / evades a disclosure obligation",
  "distinguishing": "DISTINGUISHING - what a legitimate look-alike (registered market maker, hedger, arbitrageur, genuine benchmarked order) would show here that this trader does NOT"},
 "market_open": {
  "mech": "MECHANISM - the specific abusive trading/order move (not a generic label like 'spoofing')",
  "breach": "BREACH - the specific statutory provision breached and WHY its purpose is defeated (cite SFA s.197/198/201 / MAR Art 8/12/15 / IOSCO)",
  "distinguishing": "DISTINGUISHING - what a genuine legitimate participant would show here that this one does NOT (the contrast with the innocent look-alike)"},
}
def four_points(rubric): return (rubric or "").split("ANSWER KEY")[0].strip()

def judge_elem(client, model, jkey, rubric, ek, answer):
    """Returns 1/0, or None when the judge could not be reached (never cached; counted as 0 for this run)."""
    if not (answer or "").strip(): return 0
    msg = (f"RUBRIC for this case (the correct answer):\n{four_points(rubric)}\n\n"
           f"The analyst's FULL answer:\n\"{answer[:3000]}\"\n\n"
           f"Focus ONLY on the {LABELS[jkey][ek]} element. Does the answer correctly convey THAT one element as "
           f"required by the rubric? Ignore the other elements entirely. Output ONLY 1 or 0.")
    for tokkw in ("max_tokens", "max_completion_tokens"):
        try:
            r = client.chat.completions.create(model=model, messages=[{"role": "system", "content": JSYS[jkey]}, {"role": "user", "content": msg}], **{tokkw: 2000})
            ds = re.findall(r'[01]', r.choices[0].message.content or ""); return int(ds[-1]) if ds else 0
        except Exception as e:
            if any(w in str(e).lower() for w in ("max_tokens", "max_completion", "unsupported")): continue
            time.sleep(2)
            try:
                r = client.chat.completions.create(model=model, messages=[{"role": "system", "content": JSYS[jkey]}, {"role": "user", "content": msg}], max_completion_tokens=2000)
                ds = re.findall(r'[01]', r.choices[0].message.content or ""); return int(ds[-1]) if ds else 0
            except Exception: return None
    return None

# ---------------------------------------------------------------- loading ------------------------------
def read_csv(path):
    with open(path, encoding="utf-8") as f: return list(csv.DictReader(f))

def load_frontier(domain, results_dir, outdir, arm_tag, model_tag, warnings):
    held = set(HELD[domain])
    if domain == "aml":
        p = os.path.join(outdir, f"exam_v3_{arm_tag}_{model_tag}_heldout30.csv")
        if not os.path.exists(p): p = os.path.join(results_dir, f"exam_v3_{arm_tag}_{model_tag}_heldout30.csv")
        if not os.path.exists(p):
            p = os.path.join(results_dir, f"exam_v3_{arm_tag}_{model_tag}.csv")
            warnings.append(f"{model_tag}/{arm_tag}: heldout30 file missing - using the original run (27 of 30 typologies, 4 with older story text)")
        rows = read_csv(p)
        rows = [r for r in rows if r.get("condition", "no_grounding") == "no_grounding"]
    else:
        rows = read_csv(os.path.join(results_dir, f"market100_{arm_tag}_{model_tag}.csv"))
    rows = [r for r in rows if r["typology_id"] in held]
    return rows

TRIM_RUNON = [True]    # default: judge only the first complete answer of an open model (--keep-runon disables)
def trim_runon(a):
    """Some open models (mainly Gemma-2-9B) do not stop at the end of their turn: after the five answer lines they
    write 'model' and start a second answer, sometimes with the opposite decision. Keep the text up to the second
    'DECISION:' (or a stray role line after DISTINGUISHING). This is what stopping at the end-of-turn token does in
    eval_synth_test.py; nothing inside the first answer is changed."""
    a = a or ""; i = a.find("DECISION:"); j = a.find("DECISION:", i + 9) if i >= 0 else -1
    cut = j if j > 0 else len(a)
    d = a.find("DISTINGUISHING")
    m = re.search(r"\n[ \t]*(model|user|assistant)(\.py)?[ \t]*(\n|$)", a[:cut])
    if m and d >= 0 and m.start() > d: cut = min(cut, m.start())
    return a[:cut].rstrip()
def _maybe_trim(rows):
    if TRIM_RUNON[0]:
        for r in rows:
            if "answer" in r: r["answer"] = trim_runon(r["answer"])
    return rows

def load_open(domain, results_dir, model_tag, arm):
    rows = read_csv(os.path.join(results_dir, open_file(domain, model_tag, arm)))
    return _maybe_trim([r for r in rows if r.get("split") == "unseen"])

# ---------------------------------------------------------------- scoring ------------------------------
def score(rows, domain, judge, cache, lock, pool, label):
    """rows: list of answer rows (violation + twin) for one (model, arm). Returns summary dict + per-case list."""
    V = [r for r in rows if r["instance"] == "violation"]; T = [r for r in rows if r["instance"] != "violation"]
    e2 = "loophole" if domain == "aml" else "breach"
    det = sum(1 for r in V if r["decision"] == "REPORT"); fp = sum(1 for r in T if r["decision"] == "REPORT")
    unc = sum(1 for r in V + T if r["decision"] == "UNCLEAR")
    def key(r, ek): return hashlib.sha1((domain + "|" + judge["jkey"] + "|" + r["typology_id"] + "|" + ek + "|" + (r.get("answer") or "")).encode("utf-8")).hexdigest()
    cache_local = {}
    jobs = [(r, ek) for r in V for ek in ("mech", e2, "distinguishing") if key(r, ek) not in cache]
    if jobs:
        def run(r, ek):
            return key(r, ek), judge_elem(judge["client"], judge["model"], judge["jkey"], RUB[domain].get(r["typology_id"], ""), ek, r.get("answer", ""))
        futs = [pool.submit(run, r, ek) for r, ek in jobs]
        failed = 0
        for n, fu in enumerate(as_completed(futs), 1):
            k, v = fu.result()
            if v is None: failed += 1; cache_local[k] = 0
            else:
                with lock: cache[k] = v
            if n % 15 == 0 or n == len(jobs): print(f"\r    judging {label}: {n}/{len(jobs)}      ", end="", flush=True)
        print()
        if failed:
            print(f"    WARNING: {failed}/{len(jobs)} judge calls FAILED for {label} (counted as 0, NOT cached - re-run to retry).")
            if failed == len(jobs): sys.exit("    Every judge call failed - the API is unreachable (proxy? rate limit?). Fix and re-run; nothing wrong was cached.")
    per = []; agg = {"mech": 0, e2: 0, "distinguishing": 0}; full = 0
    for r in V:
        s = {ek: cache.get(key(r, ek), cache_local.get(key(r, ek), 0)) for ek in ("mech", e2, "distinguishing")}
        dec = 1 if r["decision"] == "REPORT" else 0
        fu = 1 if (s["mech"] and s[e2] and s["distinguishing"] and dec) else 0
        full += fu
        for ek in agg: agg[ek] += s[ek]
        per.append({"typology_id": r["typology_id"], "item_id": r.get("item_id") or r.get("mid"), "mech": s["mech"], e2: s[e2],
                    "distinguishing": s["distinguishing"], "decision_correct": dec, "fully": fu, "source": r.get("source", "")})
    n = len(V) or 1
    pc = lambda x: round(100 * x / n)   # the paper's tables use integer percentages
    summ = {"n_pairs": len(V), "Det": f"{det}/{len(V)}", "FP": f"{fp}/{len(T)}", "unclear": unc,
            "Mec": pc(agg["mech"]), ("Loo" if domain == "aml" else "Bre"): pc(agg[e2]), "Dis": pc(agg["distinguishing"]), "Ful": pc(full),
            "Ful_count": f"{full}/{len(V)}"}
    return summ, per

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--results-dir", default=os.environ.get("RESULTS_DIR", HERE), help="folder with the answer CSVs")
    ap.add_argument("--outdir", default=os.environ.get("OUTDIR", HERE))
    ap.add_argument("--domains", default="aml,market")
    ap.add_argument("--include-base", action="store_true", help="also judge the open models before fine-tuning")
    ap.add_argument("--include-expert", action="store_true", help="also judge the expert-only fine-tunes (*_EXPERT_answers.csv, arm FT-expert)")
    ap.add_argument("--include-mix", action="store_true", help="also judge the synthetic+expert fine-tunes (*_MIX_answers.csv, arm FT-mix)")
    ap.add_argument("--include-repro", action="store_true", help="also judge the synthetic-only reruns (*_REPRO_answers.csv, arm FT-repro; skips models without a file)")
    ap.add_argument("--include-mixexp", action="store_true", help="also judge the synthetic + REWRITTEN expert fine-tunes (*_MIXEXP_answers.csv, arm FT-mixexp)")
    ap.add_argument("--market-labels", choices=["frontier", "open"], default="frontier")
    ap.add_argument("--workers", type=int, default=int(os.environ.get("SB_WORKERS", "4")))
    ap.add_argument("--smoke", action="store_true")
    ap.add_argument("--keep-runon", action="store_true", help="open models: judge the raw text, including anything generated after the first complete answer")
    args = ap.parse_args()
    base = os.environ.get("SB_JUDGE_BASE", "https://api.anthropic.com/v1/"); key = os.environ.get("SB_API_KEY", "")
    jmodel = os.environ.get("SB_JUDGE_MODEL", "claude-sonnet-5")
    if not key: sys.exit("Set SB_API_KEY (Anthropic key).")
    from openai import OpenAI
    client = OpenAI(base_url=base, api_key=key, timeout=90.0, max_retries=4)
    if args.smoke:
        rub = ("A correct answer must:\n1. MECHANISM — sub-threshold structuring of cash.\n2. LOOPHOLE — each deposit under the S$20,000 CDD trigger on its face but its purpose (aggregate scrutiny) is defeated by splitting.\n3. DISTINGUISHING — a genuine cash business banks its takings in one lot.\n4. DECISION — report.")
        g = judge_elem(client, jmodel, "aml", rub, "loophole", "DECISION: REPORT. Each deposit stays just under the S$20,000 CDD trigger, but the purpose (aggregate scrutiny) is defeated by splitting the cash across days; a real business banks in one lot.")
        w = judge_elem(client, jmodel, "aml", rub, "loophole", "DECISION: REPORT. This looks suspicious.")
        print(f"SMOKE strong (expect 1): {g}   weak (expect 0): {w}"); return
    os.makedirs(args.outdir, exist_ok=True)
    TRIM_RUNON[0] = not args.keep_runon; SUF = ""
    cpath = os.path.join(args.outdir, "judge_cache.json")
    cache = json.load(open(cpath)) if os.path.exists(cpath) else {}
    lock = threading.Lock(); pool = ThreadPoolExecutor(max_workers=args.workers)
    def save_cache():
        with lock: json.dump(cache, open(cpath, "w"))
    warnings, summary, percase = [], [], []
    for domain in [d for d in args.domains.split(",") if d in ("aml", "market")]:
        jkey = "aml" if domain == "aml" else f"market_{args.market_labels}"
        judge = {"client": client, "model": jmodel, "jkey": jkey}
        arms = AML_ARMS if domain == "aml" else MKT_ARMS
        print(f"\n== {domain.upper()}  held-out typologies: {len(HELD[domain])}   judge={jmodel} ({jkey})")
        for mname, mtag in OPEN:
            for arm in (((["BASE"] if args.include_base else []) + ["LORA"] + (["EXPERT"] if args.include_expert else []) + (["MIX"] if args.include_mix else []) + (["REPRO"] if args.include_repro else []) + (["MIXEXP"] if args.include_mixexp else []))):
                try: rows = load_open(domain, args.results_dir, mtag, arm)
                except FileNotFoundError as e: warnings.append(f"missing {e.filename}"); continue
                s, per = score(rows, domain, judge, cache, lock, pool, f"{mname} {arm}")
                row = {"domain": domain, "group": "open", "model": mname, "arm": {"LORA": "FT", "BASE": "Base", "EXPERT": "FT-expert", "MIX": "FT-mix", "REPRO": "FT-repro", "MIXEXP": "FT-mixexp"}[arm], **s}
                summary.append(row); percase += [{"domain": domain, "group": "open", "model": mname, "arm": row["arm"], **p} for p in per]
                print(f"  {mname:16} {row['arm']:12} Det {s['Det']:>6} FP {s['FP']:>6}  Mec {s['Mec']:3} {('Loo' if domain=='aml' else 'Bre')} {s['Loo' if domain=='aml' else 'Bre']:3} Dis {s['Dis']:3} Ful {s['Ful']:3}  ({s['Ful_count']})")
                save_cache()
        # base open models under the frontier prompt arms (cluster job eval_open_prompt_arms.py); skipped if absent
        for mname, mtag in OPEN_ARMS_MODELS:
            for aname, atag in arms:
                p = os.path.join(args.outdir, f"openbase_{domain}_{atag}_{mtag}.csv")
                if not os.path.exists(p): p = os.path.join(args.results_dir, f"openbase_{domain}_{atag}_{mtag}.csv")
                if not os.path.exists(p): continue
                rows = _maybe_trim(read_csv(p))
                s, per = score(rows, domain, judge, cache, lock, pool, f"{mname} base {aname}")
                row = {"domain": domain, "group": "open-base-prompt", "model": mname, "arm": aname, **s}
                summary.append(row); percase += [{"domain": domain, "group": "open-base-prompt", "model": mname, "arm": aname, **p} for p in per]
                print(f"  {mname:16} {aname:12} Det {s['Det']:>6} FP {s['FP']:>6}  Mec {s['Mec']:3} {('Loo' if domain=='aml' else 'Bre')} {s['Loo' if domain=='aml' else 'Bre']:3} Dis {s['Dis']:3} Ful {s['Ful']:3}  ({s['Ful_count']})")
                save_cache()
        for mname, mtag in FRONTIER:
            for aname, atag in arms:
                try: rows = load_frontier(domain, args.results_dir, args.outdir, atag, mtag, warnings)
                except FileNotFoundError as e: warnings.append(f"missing {e.filename}"); continue
                s, per = score(rows, domain, judge, cache, lock, pool, f"{mname} {aname}")
                row = {"domain": domain, "group": "frontier", "model": mname, "arm": aname, **s}
                summary.append(row); percase += [{"domain": domain, "group": "frontier", "model": mname, "arm": aname, **p} for p in per]
                print(f"  {mname:16} {aname:12} Det {s['Det']:>6} FP {s['FP']:>6}  Mec {s['Mec']:3} {('Loo' if domain=='aml' else 'Bre')} {s['Loo' if domain=='aml' else 'Bre']:3} Dis {s['Dis']:3} Ful {s['Ful']:3}  ({s['Ful_count']})")
                save_cache()
    pool.shutdown(wait=True); save_cache()
    # ---- write outputs
    cols = ["domain", "group", "model", "arm", "n_pairs", "Det", "FP", "unclear", "Mec", "Loo", "Bre", "Dis", "Ful", "Ful_count"]
    with open(os.path.join(args.outdir, "shared_heldout_summary" + SUF + ".csv"), "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=cols); w.writeheader()
        for r in summary: w.writerow({k: r.get(k, "") for k in cols})
    pcols = ["domain", "group", "model", "arm", "typology_id", "item_id", "mech", "loophole", "breach", "distinguishing", "decision_correct", "fully", "source"]
    with open(os.path.join(args.outdir, "shared_heldout_percase_grades" + SUF + ".csv"), "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=pcols); w.writeheader()
        for r in percase: w.writerow({k: r.get(k, "") for k in pcols})
    write_tex(summary, os.path.join(args.outdir, "shared_heldout_table" + SUF + ".tex"))
    print(f"\nSAVED -> {args.outdir}/shared_heldout_summary{SUF}.csv, shared_heldout_percase_grades{SUF}.csv, shared_heldout_table{SUF}.tex")
    for wmsg in warnings: print("WARNING:", wmsg)

def write_tex(summary, path):
    out = []
    for domain, e2, n in (("aml", "Loo", 30), ("market", "Bre", 25)):
        rows = [r for r in summary if r["domain"] == domain]
        if not rows: continue
        out.append(f"% {domain}: {n} held-out typologies, same cases for every row; one judge pass (Claude Sonnet 5)")
        out.append("\\begin{tabular}{llcccccc}\n\\toprule")
        out.append(f"Model & Setting & Det & FP & Mec & {e2} & Dis & Ful \\\\\n\\midrule")
        for grp, title in (("open", "Open models, fine-tuned (SFT prompt)"), ("open-base-prompt", "Open models, base, frontier prompt arms"), ("frontier", "Frontier models")):
            g = [r for r in rows if r["group"] == grp]
            if not g: continue
            out.append(f"\\multicolumn{{8}}{{l}}{{\\textit{{{title}}}}} \\\\")
            last = None
            for r in g:
                name = r["model"] if r["model"] != last else ""
                last = r["model"]
                out.append(f"{name} & {r['arm']} & {r['Det']} & {r['FP']} & {r['Mec']} & {r[e2]} & {r['Dis']} & {r['Ful']} \\\\")
        out.append("\\bottomrule\n\\end{tabular}\n")
    open(path, "w", encoding="utf-8").write("\n".join(out))

if __name__ == "__main__":
    main()
