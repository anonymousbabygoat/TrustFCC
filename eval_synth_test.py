# -*- coding: utf-8 -*-
"""
Evaluate an ALREADY FINE-TUNED open model (base + LoRA) on the synthetic held-out TEST split.

Same prompt, decoding and output schema as eval_aml90_2way.py / eval_market_2way.py (the expert-test-set scripts),
so the existing graders read the output unchanged. Differences: (1) cases come from test_items_<domain>.json
(the CLEAN synthetic split: 889 AML / 721 market pairs from the 55 held-out typologies) instead of the 120/100 expert
seeds; (2) generation is batched (BATCH, default 8, left padding) because there are ~1,800 cases per model.

Env: DOMAIN=aml|market, BASE_MODEL, LORA_DIR, USE_LORA=1, OUT_CSV, MAX_NEW (512 aml / 1024 market), BATCH, ITEMS, LIMIT
Output columns: mid,typology_id,family,split,instance,gold_decision,decision,passed,answer   (split = "synthetic_test")
"""
import os, re, csv, json, time, torch
os.environ.setdefault('HF_HUB_OFFLINE','1'); os.environ.setdefault('TRANSFORMERS_OFFLINE','1')
os.environ.setdefault('TOKENIZERS_PARALLELISM','false')
from transformers import AutoTokenizer, AutoModelForCausalLM

DOMAIN  = os.environ.get("DOMAIN", "aml")
ITEMS   = json.load(open(os.environ.get("ITEMS", f"test_items_{DOMAIN}.json"), encoding="utf-8"))
MODEL   = os.environ['BASE_MODEL']
LORA    = os.environ.get('LORA_DIR', '')
USE_LORA= os.environ.get('USE_LORA','1')=='1'
OUT     = os.environ.get('OUT_CSV', f'synthtest_{DOMAIN}_answers.csv')
MAXNEW  = int(os.environ.get('MAX_NEW', '512' if DOMAIN=='aml' else '1024'))
BATCH   = int(os.environ.get('BATCH', '8'))
LIMIT   = int(os.environ.get('LIMIT', '0'))
SHARD   = os.environ.get('SHARD', '')          # "k/N": handle items k, k+N, k+2N, ... (split one model over several GPUs)

if DOMAIN == "aml":   # EXACT SFT prompt (eval_aml90_2way.py)
    SYSTEM = ("You are an experienced AML/CFT analyst at a bank regulated in Singapore. You are given a customer's "
              "profile and transaction record and must decide whether the activity should be REPORTED to compliance as "
              "suspicious (for a possible STR), or is legitimate and can PROCEED. Judge by the substance and PURPOSE of "
              "the AML rules (MAS Notice 626; CDSA; FATF standards), not merely whether a rule is technically satisfied "
              "on its face. When you report, give a complete, auditable justification.")
    INSTR = ("\n\nDecide whether this should be REPORTED to compliance as suspicious, or can PROCEED as legitimate.\n"
             "Answer in EXACTLY these lines, nothing before them:\n"
             "DECISION: REPORT or PROCEED\n"
             "TYPOLOGY: the specific laundering typology (or NONE if legitimate)\n"
             "MECHANISM: the specific technical move the customer is using\n"
             "LOOPHOLE: which rule/threshold/exemption is met on its face and how its purpose is defeated (cite MAS 626 / CDSA)\n"
             "DISTINGUISHING: the specific fact(s) that separate this from a genuine, legitimate look-alike")
    GOLD_OK = "PROCEED"
    def parse_decision(text):
        tl=(text or "").lower()
        m=re.search(r'decision\s*:\s*(report|proceed)', tl)
        if m: return m.group(1).upper()
        rep=any(w in tl for w in ("report","flag","suspicious","escalate","file a str","file an str"," sar"))
        proc=any(w in tl for w in ("proceed","legitimate","not suspicious","no action","clear"))
        if rep and not proc: return "REPORT"
        if proc and not rep: return "PROCEED"
        return "UNCLEAR"
else:                 # EXACT SFT prompt (eval_market_2way.py)
    SYSTEM = ("You are an experienced market surveillance analyst at a firm regulated in Singapore. You are given a "
              "trading and order record and must decide whether the activity should be REPORTED to surveillance as market "
              "abuse (for escalation), or is legitimate and can be CLEARED. Judge by the substance and PURPOSE of the "
              "market-abuse rules (SFA Part XII; EU MAR; IOSCO), not merely whether a rule is technically satisfied on its "
              "face. When you report, give a complete, auditable justification.")
    INSTR = ("\n\nDecide whether this should be REPORTED to surveillance as market abuse, or can be CLEARED as legitimate.\n"
             "Answer in EXACTLY these lines, nothing before them:\n"
             "DECISION: REPORT or CLEAR\n"
             "TYPOLOGY: the specific market-abuse typology (or NONE if legitimate)\n"
             "MECHANISM: the specific abusive move the trader is using\n"
             "BREACH: which statutory provision is breached and why (cite SFA s.197/198/201 / MAR Art 8/12/15 / IOSCO)\n"
             "DISTINGUISHING: the specific fact(s) that separate this from a genuine, legitimate look-alike")
    GOLD_OK = "CLEAR"
    def parse_decision(text):
        tl=(text or "").lower()
        m=re.search(r'decision\s*:\s*(report|clear)', tl)
        if m: return m.group(1).upper()
        rep=any(w in tl for w in ("report","flag","escalate","suspicious","manipulat","abuse"))
        clr=any(w in tl for w in ("clear","legitimate","not abuse","no action","bona fide"))
        if rep and not clr: return "REPORT"
        if clr and not rep: return "CLEAR"
        return "UNCLEAR"

CASES=[]
_items = ITEMS[:LIMIT] if LIMIT else ITEMS
if SHARD:
    k, N = (int(x) for x in SHARD.split("/")); _items = _items[k::N]
for s in _items:
    CASES.append({"mid":s["mid"]+"-violation","typology_id":s["seed_id"],"family":s["family"],"instance":"violation","gold":"REPORT","story":s["vstory"]})
    CASES.append({"mid":s["mid"]+"-benign_twin","typology_id":s["seed_id"],"family":s["family"],"instance":"benign_twin","gold":GOLD_OK,"story":s["tstory"]})

dtype = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
tok = AutoTokenizer.from_pretrained(MODEL, trust_remote_code=True, use_fast=True)
if tok.pad_token is None: tok.pad_token = tok.eos_token
tok.padding_side = "left"
model = AutoModelForCausalLM.from_pretrained(MODEL, torch_dtype=dtype, trust_remote_code=True, device_map="auto")
if USE_LORA:
    from peft import PeftModel
    model = PeftModel.from_pretrained(model, LORA); print("loaded LoRA:", LORA)
    if os.environ.get("MERGE_LORA", "0") == "1":   # fold the adapter into the weights (same model, no fp32 adapter overhead)
        model = model.merge_and_unload(); print("LoRA merged into base weights")
    model = model.to(dtype)
model.eval()
# Stop at the chat-template end-of-turn token as well as EOS (Gemma-2 otherwise runs to the token cap writing a
# hallucinated next turn after its answer; the graded five lines are unchanged, only the trailing junk is cut).
EOS_IDS = [tok.eos_token_id]
for _t in ("<end_of_turn>", "<|eot_id|>", "<|im_end|>"):
    _i = tok.convert_tokens_to_ids(_t)
    if isinstance(_i, int) and _i >= 0 and _i != tok.unk_token_id and _i not in EOS_IDS: EOS_IDS.append(_i)
print("eos ids:", EOS_IDS)
print(f"{DOMAIN} synthetic test: {len(CASES)} cases ({len(CASES)//2} pairs) | LoRA={USE_LORA} | max_new={MAXNEW} | batch={BATCH}", flush=True)

def build_prompt(story):
    msgs=[{"role":"system","content":SYSTEM},{"role":"user","content":story.strip()+INSTR}]
    try:
        return tok.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)
    except Exception:   # no 'system' role (Gemma-2, some Mistral): fold system into the user turn
        return tok.apply_chat_template([{"role":"user","content":SYSTEM+"\n\n"+story.strip()+INSTR}], tokenize=False, add_generation_prompt=True)

# resume: keep rows already written
done={}
if os.path.exists(OUT):
    for r in csv.DictReader(open(OUT, encoding="utf-8")):
        if r.get("answer","").strip(): done[r["mid"]]=r
    print(f"[resume] {len(done)} rows kept")
rows=[done[c["mid"]] for c in CASES if c["mid"] in done]
todo=[c for c in CASES if c["mid"] not in done]
FIELDS=["mid","typology_id","family","split","instance","gold_decision","decision","passed","answer"]
def flush():
    ordered={r["mid"]:r for r in rows}
    with open(OUT,"w",newline="",encoding="utf-8") as f:
        w=csv.DictWriter(f,fieldnames=FIELDS); w.writeheader(); w.writerows([ordered[c["mid"]] for c in CASES if c["mid"] in ordered])
t0=time.time()
for b in range(0, len(todo), BATCH):
    chunk=todo[b:b+BATCH]
    prompts=[build_prompt(c["story"]) for c in chunk]
    enc=tok(prompts, return_tensors="pt", padding=True).to(model.device)
    with torch.no_grad():
        out=model.generate(**enc, max_new_tokens=MAXNEW, do_sample=False, pad_token_id=tok.pad_token_id, eos_token_id=EOS_IDS)
    for i,c in enumerate(chunk):
        ans=tok.decode(out[i][enc["input_ids"].shape[1]:], skip_special_tokens=True).strip()
        dec=parse_decision(ans)
        rows.append({"mid":c["mid"],"typology_id":c["typology_id"],"family":c["family"],"split":"synthetic_test",
                     "instance":c["instance"],"gold_decision":c["gold"],"decision":dec,
                     "passed":"PASS" if dec==c["gold"] else "FAIL","answer":ans})
    n=len(rows)
    if (b//BATCH)%5==0 or b+BATCH>=len(todo):
        flush(); print(f"  {n}/{len(CASES)}  ({time.time()-t0:.0f}s)", flush=True)
flush()
V=[r for r in rows if r["instance"]=="violation"]; T=[r for r in rows if r["instance"]!="violation"]
print(f"\n== decision-only == detection {sum(1 for r in V if r['decision']=='REPORT')}/{len(V)}  "
      f"false-pos {sum(1 for r in T if r['decision']=='REPORT')}/{len(T)}  unclear {sum(1 for r in rows if r['decision']=='UNCLEAR')}")
print("wrote ->", OUT)
