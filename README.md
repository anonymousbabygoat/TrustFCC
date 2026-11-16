# TrustFCC: Benchmarking Reporting Decisions and Explanations in Financial-Crime Compliance

Code for a matched-twin benchmark of **auditable explanations** across anti-money-laundering (AML) and market abuse. Each scenario is scored against a four-element rubric — **mechanism**, the **rule that is defeated** (loophole in AML, breach in market abuse), the **distinguishing** facts that separate a violation from its innocent twin, and the **decision** (REPORT / PROCEED / CLEAR) — rather than a yes/no label.

*Anonymous repository for peer review. It contains no real customer, transaction, or account data; every scenario is synthetic.*

The dataset (10,057 matched pairs = 20,114 scenarios, plus expert seeds) is released separately: **https://huggingface.co/datasets/Anonymousbabygoat/TrustFCC**.

## Pipeline

1. **Expert seeds.** A qualified Singapore compliance lawyer authored one matched pair per typology — a *violation* and an *innocent twin* — with a four-element answer key for each. `seeds_aml.json` (120 typologies: 90 seen / 30 held-out) and `seeds_market.json` (100: 75 / 25). Each record has `id, family, typology, split, vstory, vrubric, tstory, trubric`.

2. **Scaling with automatic checks.** `scaler_aml.py` / `scaler_market.py` few-shot a teacher model (Claude Sonnet 5) from each **training** seed to generate new matched pairs on rotating diversity axes, and run automatic checks on every generated pair:
   - schema completeness (all four elements present);
   - a **give-away-word guard** on the story bodies (`LEAK_WORDS`), so the neutral record never names the abuse;
   - a **near-duplicate guard** — flag if string similarity to the seed `> 0.85`, or to an earlier variant of the same typology `> 0.75`.
   Held-out typologies are never generated. Flagged rows are regenerated (`SC_REDO_FLAGGED=1`). The API key is read from the environment and never stored.

   ```bash
   export SC_API_KEY=...            # teacher-model key (not stored)
   export SC_BASE=https://api.anthropic.com/v1/   SC_MODEL=claude-sonnet-5
   export SC_SEEDS=seeds_aml.json   SC_A_PER=56   SC_OUT=AML90_scaled_FULL.csv
   python3 scaler_aml.py            # market: scaler_market.py, SC_SEEDS=seeds_market.json, SC_A_PER=67
   ```

3. **Fine-tuning.** `train_aml_sft.py` / `train_market_sft.py` train a LoRA adapter (rank 16, α=32, dropout 0.05) on the attention and MLP projections for three epochs with frozen base weights (transformers + peft only). PBS launchers (`run_aml_sft.pbs`, `run_market_eval_qwen7b.pbs`, `download_models.pbs`); the SFT target is built from the rubric so the model learns to emit the four elements.

4. **Evaluation.** `eval_aml90_2way.py` / `eval_market_2way.py` run base vs. LoRA on the **held-out** typologies (two-way: violation + innocent twin), writing per-scenario answers to CSV. Seeds are loaded from the JSON files (`EVAL_SEEDS`).

5. **Grading.** Each reasoning element is graded in its own call by a judge model (Claude Sonnet 5), with the decision checked directly against the answer key; a scenario is *fully correct* only when all three reasoning elements pass and the decision is right (`grade_aml_2way.py` with `grade_rubrics_aml.json`; `grade_market_2way.py` with `grade_rubrics_market.json`, which sweeps all fine-tuned models).

6. **Prompting frontier models + APO.** An innocent-first prompt (Contrast in AML, Innocent-Hypothesis-First in market abuse) builds the innocent twin before naming the distinguishing facts; automatic prompt optimization (DSPy MIPROv2) is run from a bare seed as a comparison (`dspy_apo_aml.py`, `dspy_apo_market.py`).

## Training-data settings and the larger test set

**Training-data settings.** Four settings share the LoRA recipe, validation file, test cases and judge, and differ only in the training file:

| Setting | Training file | Built by | Launcher | Answer files |
|---|---|---|---|---|
| Synthetic | `aml90_sft_train.jsonl`, `market75_sft_train.jsonl` | scalers + SFT builders | `run_aml_sft.pbs` | `*_LORA_answers.csv` |
| Expert Seeds | `aml_expert_sft_train.jsonl` (180 records), `market_expert_sft_train.jsonl` (150) | `build_expert_sft.py` | `run_expert_aml.pbs`, `run_expert_market.pbs` | `*_EXPERT_answers.csv` |
| Synthetic + Seeds | `aml_mix_sft_train.jsonl` (9,874), `market_mix_sft_train.jsonl` (9,666) | `build_mix_sft.py` | `run_mix_aml.pbs`, `run_mix_market.pbs` | `*_MIX_answers.csv` |
| Synthetic + Rewritten (AML) | `aml_mixexp_sft_train.jsonl` (9,874) | `rewrite_expert_targets.py`, then `build_mixexp_sft.py` | `run_mixexp_aml.pbs` | `*_MIXEXP_answers.csv` |

`rewrite_expert_targets.py` rewrites the 180 AML seed answers at the length and detail of the synthetic answers with Claude Sonnet 5, keeping every decision, typology and fact (checks: same DECISION and TYPOLOGY lines, five answer lines, 700 to 1,800 characters; up to three attempts, otherwise the original answer is kept). The rewritten answers are released with the dataset (`aml_seed_answers_rewritten.jsonl`). `patch_eval_split.py` adds the `EVAL_SPLIT=unseen` switch used by the launchers.

**Grading the expert test set (all settings, one judge pass).** `rescore_shared_heldout.py` scores every open-model setting and every frontier prompt on the same 30 AML and 25 market-abuse expert pairs (`heldout_ids.json`, `rubrics_aml.json`, `rubrics_market.json`). The judge prompt is the one reproduced in the paper's appendix. Open-model answers are judged on their first complete answer: text generated after the end of the answer is cut before judging (`--keep-runon` judges the raw text).

```bash
export SB_API_KEY=...   # judge key (not stored)
python3 rescore_shared_heldout.py --results-dir <folder with the answer CSVs> --outdir out \
        --include-base --include-expert --include-mix --include-mixexp
```

**Larger synthetic test set (889 AML and 721 market-abuse pairs).** The pairs are generated from the 55 held-out typologies with the same scalers (`SC_SPLIT=unseen`) and released with the dataset (`aml_test.csv.gz`, `market_test.csv.gz`).

```bash
python3 make_test_items.py                                   # csv.gz -> test_items_<domain>.json (case text only)
DOMAIN=aml BASE_MODEL=... LORA_DIR=... python eval_synth_test.py     # fine-tuned open models (run_synthtest_qwen7b.pbs)
python3 run_frontier_synth_test.py --model deepseek          # frontier reference, Baseline prompt
python3 grade_synth_test.py                                  # same judge; writes synthtest_summary.csv
```

`grade_synth_test.py` uses each pair's generated reference explanation as the rubric. For market abuse, the generated list of decoys is removed from the rubric before judging (`--rubric paper`, the default), so the judge sees the same content as on the expert test set.


## Contents

| File | Purpose |
|---|---|
| `scaler_aml.py`, `scaler_market.py` | Teacher-model scaling + automatic checks |
| `seeds_aml.json`, `seeds_market.json` | Expert seed pairs (with train/held-out split) |
| `train_aml_sft.py`, `train_market_sft.py` | LoRA SFT |
| `eval_aml90_2way.py`, `eval_market_2way.py` | Two-way held-out evaluation |
| `grade_aml_2way.py` | Per-element judge grader (base vs LoRA); loads `grade_rubrics_aml.json` |
| `grade_market_2way.py` | Per-element judge grader, all models (base vs LoRA); loads `grade_rubrics_market.json` |
| `dspy_apo_aml.py`, `dspy_apo_market.py` | Automatic prompt optimization (DSPy MIPROv2); load `apo_bundle_*`/`apo_rubrics_*` |
| `run_*.pbs`, `download_models.pbs` | Cluster launchers (representative) |
| `build_expert_sft.py`, `build_mix_sft.py`, `build_mixexp_sft.py`, `rewrite_expert_targets.py` | Training files for Expert Seeds, Synthetic + Seeds, Synthetic + Rewritten |
| `synthetic_train_sample*.jsonl` | Style examples shown to the rewriter |
| `run_expert_*.pbs`, `run_mix_*.pbs`, `run_mixexp_aml.pbs`, `run_synthtest_qwen7b.pbs`, `patch_eval_split.py` | Cluster launchers (representative) |
| `rescore_shared_heldout.py`, `heldout_ids.json`, `rubrics_*.json` | Expert-test-set grader, all settings |
| `make_test_items.py`, `eval_synth_test.py`, `run_frontier_synth_test.py`, `grade_synth_test.py` | Larger synthetic test set: evaluation and grading |

## Requirements

Generation/eval harness: `openai` (used as the HTTP client for the teacher/judge). Fine-tuning: `torch`, `transformers`, `peft`, `accelerate`. Python 3.10+.

## Ethics

The benchmark is built only from synthetic, expert-authored and model-grown scenarios; suspicious-activity and suspicious-transaction reports and the underlying records are confidential by law and are never used. The models and data are intended to assist a qualified analyst, not to file reports autonomously.
