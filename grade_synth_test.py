# -*- coding: utf-8 -*-
"""
Grade fine-tuned open models on the SYNTHETIC held-out test split with the same judge as the expert test set:
Claude Sonnet 5, one call per element, whole answer shown, decision checked directly against the label.
Judge system prompt and element labels are copied verbatim from rescore_shared_heldout.py (which copied them from the
original graders). The only difference from the expert-pair graders is the lookup: each synthetic pair carries its own
rubric (violation_rubric in the test csv), which is used as the reference.

--rubric paper (default) reproduces the paper's larger-test-set table: AML uses the pair rubric as is ("pair"); market
abuse uses the pair rubric with the generated decoy list removed ("pair_nodecoy"), so the judge sees the same kind of
rubric as on the expert test set, where the decoy list sits after the ANSWER KEY line and is never shown to the judge.
Other values: pair (rubric as generated), pair_nodecoy (decoys removed in both domains; no AML rubric contains a decoy
list, so AML is unchanged), expert (the parent typology's expert rubric).

Inputs : synthtest_<domain>_<tag>_LORA_answers.csv (eval_synth_test.py) or ..._BASE_answers.csv (run_frontier_synth_test.py),
         aml_test.csv.gz / market_test.csv.gz (rubrics), seeds_aml.json / seeds_market.json
Outputs: synthtest_summary.csv (one row per model x domain), synthtest_percase_grades.csv, judge_cache_synth.json
"""
import os, re, sys, csv, json, time, argparse, hashlib, threading, glob
csv.field_size_limit(10**9)
from concurrent.futures import ThreadPoolExecutor, as_completed
try:
    import truststore; truststore.inject_into_ssl()
except Exception:
    pass
HERE = os.path.dirname(os.path.abspath(__file__))

JSYS = {
 "aml": ("You are a strict Singapore AML examiner. You are shown the correct rubric and an analyst's full answer, and "
         "asked about ONE element only. Judge only that element; ignore whether the other elements are present. A vague "
         "or generic statement does not earn it."),
 "market": ("You are a strict Singapore market-surveillance examiner. You are shown the correct rubric and an analyst's "
         "full answer, and asked about ONE element only. Judge only that element; ignore the others. A vague or "
         "generic statement does not earn it."),
}
LABELS = {
 "aml": {
  "mech": "MECHANISM — the specific laundering method / technical move (not a generic label like 'structuring')",
  "loophole": "LOOPHOLE — a determinate rule/threshold/exemption/definition/timing/document satisfied on its face, but its PURPOSE defeated",
  "distinguishing": "DISTINGUISHING — what a genuine legitimate customer would show here that this customer does NOT (the contrast with the innocent look-alike)"},
 "market": {
  "mech": "MECHANISM - the specific market-abuse method (e.g. spoofing, layering, wash trading, marking the close), not a generic label like 'manipulation'",
  "breach": "BREACH - why the conduct breaches market-conduct rules: it creates a FALSE or MISLEADING appearance of the market, OR misuses material non-public information / evades a disclosure obligation",
  "distinguishing": "DISTINGUISHING - what a legitimate look-alike (registered market maker, hedger, arbitrageur, genuine benchmarked order) would show here that this trader does NOT"},
}
def four_points(rubric): return (rubric or "").split("ANSWER KEY")[0].strip()

# --rubric pair_nodecoy: the expert rubrics keep their decoy list inside the ANSWER KEY, which four_points() hides from
# the judge; the synthetic rubrics were generated with the decoys written inline in point 3. Remove that text so the
# judge sees the same kind of rubric on both test sets (the paper's DISTINGUISHING element does not require decoys).
_DECOY_KW = re.compile(r"decoy|must be cleared|be cleared|near-miss|not flagged|clear(ed)? for the right reason", re.I)
def strip_decoys(rubric):
    t = four_points(rubric)
    t = re.sub(r"\s*(Decoys?\b[^.:\n]{0,80}:|Clear the decoys[^:\n]{0,80}:).*?(?=\n\s*4\.|\n\s*DECISION|\Z)", "", t, flags=re.S | re.I)
    m = re.search(r"(3\.\s*DISTINGUISHING.*?)(?=\n\s*4\.|\Z)", t, re.S | re.I)
    if m:
        parts = re.split(r"(?=[;,.]?\s*\([a-h]\)\s)", m.group(1))
        keep = [x for x in parts if not _DECOY_KW.search(x)]
        if len(keep) < len(parts) and keep and len("".join(keep)) >= max(200, 0.4 * len(m.group(1))):
            t = t[:m.start()] + re.sub(r"[;,\s]+$", "", "".join(keep)) + "." + t[m.end():]
    m = re.search(r"(3\.\s*(?:Cite the )?DISTINGUISHING.*?)(?=\n\s*4\.|\Z)", t, re.S | re.I)
    if m and _DECOY_KW.search(m.group(1)):   # decoy sentences not in lettered sub-clauses: drop sentence by sentence
        sents = re.split(r"(?<=[.;])\s+(?=[A-Z(])", m.group(1))
        keep = [x for x in sents if not _DECOY_KW.search(x)]
        if keep and len(keep) < len(sents) and len(" ".join(keep)) >= max(200, 0.4 * len(m.group(1))):
            t = t[:m.start()] + re.sub(r"[;,\s]+$", "", " ".join(keep)) + "." + t[m.end():]
    return t

def judge_elem(client, model, dom, rubric, ek, answer):
    """1/0, or None if the judge could not be reached (never cached)."""
    if not (answer or "").strip(): return 0
    msg = (f"RUBRIC for this case (the correct answer):\n{four_points(rubric)}\n\n"
           f"The analyst's FULL answer:\n\"{answer[:3000]}\"\n\n"
           f"Focus ONLY on the {LABELS[dom][ek]} element. Does the answer correctly convey THAT one element as "
           f"required by the rubric? Ignore the other elements entirely. Output ONLY 1 or 0.")
    for tokkw in ("max_tokens", "max_completion_tokens"):
        try:
            r = client.chat.completions.create(model=model, messages=[{"role": "system", "content": JSYS[dom]}, {"role": "user", "content": msg}], **{tokkw: 2000})
            ds = re.findall(r'[01]', r.choices[0].message.content or ""); return int(ds[-1]) if ds else 0
        except Exception as e:
            if any(w in str(e).lower() for w in ("max_tokens", "max_completion", "unsupported")): continue
            time.sleep(2)
            try:
                r = client.chat.completions.create(model=model, messages=[{"role": "system", "content": JSYS[dom]}, {"role": "user", "content": msg}], max_completion_tokens=2000)
                ds = re.findall(r'[01]', r.choices[0].message.content or ""); return int(ds[-1]) if ds else 0
            except Exception: return None
    return None

MODELS = [("Qwen2.5-3B","qwen3b"),("Qwen2.5-7B","qwen7b"),("Llama-3.2-3B","llama3b"),("Llama-3.1-8B","llama8b"),
          ("Gemma-2-2B","gemma2b"),("Gemma-2-9B","gemma9b"),("Mistral-7B","mistral7b")]
FRONTIER = [("DeepSeek V4 Pro", "deepseek"), ("GPT-5.4", "gpt"), ("Claude Sonnet 5", "sonnet")]   # Baseline prompt, files *_BASE_answers.csv
CLEAN = {"aml": "aml_test.csv.gz", "market": "market_test.csv.gz"}   # the larger synthetic test set (dataset release)
def read_rows(path):
    import gzip
    f = gzip.open(path, "rt", encoding="utf-8", newline="") if path.endswith(".gz") else open(path, encoding="utf-8", newline="")
    with f: return list(csv.DictReader(f))
SEEDS = {"aml": "seeds_aml.json", "market": "seeds_market.json"}

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dir", default=HERE)
    ap.add_argument("--domains", default="aml,market")
    ap.add_argument("--rubric", choices=["paper", "pair", "expert", "pair_nodecoy"], default="paper")
    ap.add_argument("--models", default="", help="comma-separated model tags to judge (default all), e.g. deepseek")
    ap.add_argument("--workers", type=int, default=int(os.environ.get("SB_WORKERS", "6")))
    ap.add_argument("--smoke", action="store_true")
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
    cpath = os.path.join(args.dir, "judge_cache_synth.json")
    cache = json.load(open(cpath)) if os.path.exists(cpath) else {}
    lock = threading.Lock(); pool = ThreadPoolExecutor(max_workers=args.workers)
    def save():
        with lock: json.dump(cache, open(cpath, "w"))
    summary, percase = [], []
    spath = os.path.join(args.dir, "synthtest_summary.csv"); ppath = os.path.join(args.dir, "synthtest_percase_grades.csv")
    if os.path.exists(spath): summary = list(csv.DictReader(open(spath, encoding="utf-8")))
    if os.path.exists(ppath): percase = list(csv.DictReader(open(ppath, encoding="utf-8")))
    for domain in [d for d in args.domains.split(",") if d in CLEAN]:
        e2 = "loophole" if domain == "aml" else "breach"
        rubric_mode = args.rubric if args.rubric != "paper" else ("pair" if domain == "aml" else "pair_nodecoy")
        cp = os.path.join(args.dir, CLEAN[domain])
        if not os.path.exists(cp): print(f"[{domain}] {CLEAN[domain]} not found - skipped"); continue
        pair_rub = {r["mid"]: r["violation_rubric"] for r in read_rows(cp)}
        expert_rub = {s["id"]: s["vrubric"] for s in json.load(open(os.path.join(args.dir, SEEDS[domain]), encoding="utf-8"))}
        for mname, mtag, suf in [(a, b, "LORA") for a, b in MODELS] + [(a, b, "BASE") for a, b in FRONTIER]:
            if args.models and mtag not in args.models.split(","): continue
            f = os.path.join(args.dir, f"synthtest_{domain}_{mtag}_{suf}_answers.csv")
            shards = sorted(glob.glob(os.path.join(args.dir, f"synthtest_{domain}_{mtag}_{suf}_answers_s*of*.csv")))
            if not os.path.exists(f) and not shards: continue
            rows = []
            for fp in ([f] if os.path.exists(f) else []) + shards:
                rows += list(csv.DictReader(open(fp, encoding="utf-8")))
            rows = list({r["mid"]: r for r in rows}.values())   # merge shards, de-duplicate by case id
            V = [r for r in rows if r["instance"] == "violation"]; T = [r for r in rows if r["instance"] != "violation"]
            n_expected = 2 * len(pair_rub)
            if len(rows) < n_expected: print(f"  {mname:13} {domain}: {len(rows)}/{n_expected} rows - run not finished, skipped"); continue
            def rub_for(r):
                pid = r["mid"].rsplit("-", 1)[0]
                if rubric_mode == "expert": return expert_rub.get(r["typology_id"], "")
                if rubric_mode == "pair_nodecoy": return strip_decoys(pair_rub.get(pid, ""))
                return pair_rub.get(pid, "")
            def key(r, ek): return hashlib.sha1((domain + "|" + rubric_mode + "|" + r["mid"] + "|" + ek + "|" + (r.get("answer") or "")).encode("utf-8")).hexdigest()
            jobs = [(r, ek) for r in V for ek in ("mech", e2, "distinguishing") if key(r, ek) not in cache]
            local = {}; failed = 0
            if jobs:
                futs = [pool.submit(lambda r, ek: (key(r, ek), judge_elem(client, jmodel, domain, rub_for(r), ek, r.get("answer", ""))), r, ek) for r, ek in jobs]
                for n, fu in enumerate(as_completed(futs), 1):
                    k, v = fu.result()
                    if v is None: failed += 1; local[k] = 0
                    else:
                        with lock: cache[k] = v
                    if n % 50 == 0 or n == len(jobs):
                        print(f"\r    judging {mname} {domain}: {n}/{len(jobs)}      ", end="", flush=True)
                        if n % 300 == 0: save()
                print()
                if failed == len(jobs): sys.exit("    every judge call failed - API unreachable (proxy? rate limit?). Nothing cached; re-run.")
                if failed: print(f"    WARNING: {failed} judge calls failed (counted 0, not cached) - re-run to retry them.")
            agg = {"mech": 0, e2: 0, "distinguishing": 0}; full = 0
            percase = [p for p in percase if not (p["domain"] == domain and p["model"] == mname and p["rubric"] == rubric_mode)]
            for r in V:
                s = {ek: cache.get(key(r, ek), local.get(key(r, ek), 0)) for ek in ("mech", e2, "distinguishing")}
                dec = 1 if r["decision"] == "REPORT" else 0
                fu = 1 if (s["mech"] and s[e2] and s["distinguishing"] and dec) else 0; full += fu
                for ek in agg: agg[ek] += s[ek]
                percase.append({"domain": domain, "model": mname, "rubric": rubric_mode, "mid": r["mid"], "typology_id": r["typology_id"],
                                "mech": s["mech"], e2: s[e2], "distinguishing": s["distinguishing"], "decision_correct": dec, "fully": fu})
            nV = len(V); det = sum(1 for r in V if r["decision"] == "REPORT"); fp = sum(1 for r in T if r["decision"] == "REPORT")
            unc = sum(1 for r in rows if r["decision"] == "UNCLEAR")
            row = {"domain": domain, "model": mname, "rubric": rubric_mode, "n_pairs": nV, "Det": f"{det}/{nV}", "FP": f"{fp}/{len(T)}", "unclear": unc,
                   "Det_rate": round(det / nV, 3), "FP_rate": round(fp / len(T), 3), "Mec": round(agg["mech"] / nV, 3),
                   ("Loo" if domain == "aml" else "Bre"): round(agg[e2] / nV, 3), "Dis": round(agg["distinguishing"] / nV, 3), "FCR": round(full / nV, 3)}
            summary = [s for s in summary if not (s["domain"] == domain and s["model"] == mname and s["rubric"] == rubric_mode)] + [row]
            print(f"  {mname:13} {domain:6} Det {row['Det']:>8} FP {row['FP']:>8}  Mec {row['Mec']:.3f} {('Loo' if domain=='aml' else 'Bre')} {row['Loo' if domain=='aml' else 'Bre']:.3f} Dis {row['Dis']:.3f} FCR {row['FCR']:.3f}")
            save()
    pool.shutdown(wait=True); save()
    cols = ["domain", "model", "rubric", "n_pairs", "Det", "FP", "unclear", "Det_rate", "FP_rate", "Mec", "Loo", "Bre", "Dis", "FCR"]
    with open(spath, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=cols); w.writeheader()
        for r in summary: w.writerow({k: r.get(k, "") for k in cols})
    pcols = ["domain", "model", "rubric", "mid", "typology_id", "mech", "loophole", "breach", "distinguishing", "decision_correct", "fully"]
    with open(ppath, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=pcols); w.writeheader()
        for r in percase: w.writerow({k: r.get(k, "") for k in pcols})
    print(f"\nSAVED -> {spath}\n         {ppath}")

if __name__ == "__main__":
    main()
