# Brief: set up the mMICE-P training pipeline (branch `multilingual-mice`, repo `xpmir/mice`)

## Standing rules (apply to every step)
- Work only on the current local branch `multilingual-mice`. Never push, never touch the remote.
- Ask for my permission before making any change to the repository. Propose the change (files, diff summary), wait for approval, then apply it.
- Never commit before I have reviewed the code. If I later ask for a commit, end the message with the attribution lines required by your environment.
- Respect the experimaestro framework and keep the code consistent with its current state (naming, `@configuration()` classes, `GridSearch` fields, YAML inheritance). Prefer adding new fields/branches over modifying existing behaviour, so the English experiments still run unchanged.
- Do not launch training or evaluation runs. Dry runs only (`uv run experimaestro run <config>.yaml --run-mode DRY_RUN`). I run the real jobs.
- Stop after each step and report. Do not chain steps without my go-ahead.
- If something below turns out to be false in the code, tell me instead of working around it.

## Goal
Train **mMICE-P-130m**: a multilingual MICE-P model that follows the actual MICE-P recipe (paper: arXiv 2610.31932, section 2.2), but with multilingual fine-tuning data. This is done in three experiments:
- **E0**: measure the first-stage ceiling (no training).
- **E4**: train a multilingual cross-encoder reranker. It is the starting checkpoint for E5 (the multilingual equivalent of `cross-encoder/ettin-reranker-*-v1`).
- **E5**: train mMICE-P from the E4 checkpoint, in two phases, exactly as the existing MICE-P configs do.

## Decisions already taken
- Languages for training: **EN, FR, ES, IT, DE, PT**.
- Training data: `lightonai/embeddings-fine-tuning-filtered-{en,fr,es,it,de,pt}`, **translated splits only**: fiqa, hotpotqa, nq, msmarco, fever, squadv2, trivia (7 sources x 6 languages = 42 splits). **Exclude `miracl_*` and `mldr_*`** (these are evaluation sets: training on them would leak).
- Teacher signal: the `rerank_scores` field (cross-encoder `mxbai-rerank-large-v2`), not the bi-encoder `scores`. This is the same teacher as MICE-P. Loss: pointwise MSE, as in MICE-P.
- Batches follow **LightOn's recipe**, not language-balanced batches: each batch is drawn entirely from one split, the split being chosen per batch with probability proportional to `len(split) ** alpha`, default `alpha = 1.0`. Reference implementation: `MultinomialBatchSampler` in `lightonai/mdenseon-mlateon`, `scripts/finetune/multilingual_dense.py` (around lines 240-285; exhausted splits are restarted with a reseeded sampler).
- First stage for all evaluations: `naver/splade-v3-distilbert`, top-100, as in `experiments/micep/_common_eval.yaml`.
- Evaluation: MIRACL on **EN, ES, FR, DE**; MLDR on **EN, ES, FR, DE, IT, PT** (IT and PT only exist in MLDR).
- Lengths: **inherit the MICE-P values unchanged, do not override them**. Standard re-ranking uses `max_query_length: 1024` / `max_doc_length: 1024`; the `plaid:` block uses 32 / 512. MLDR may suffer from this truncation, which is accepted.
- Validation set for checkpoint selection: the multilingual NanoBEIR released by LightOn (`lightonai/nanobeir-multilingual`), restricted to the six languages. It must not include MIRACL or MLDR.
- Starting backbone for E4: `jhu-clsp/mmBERT-small`.
- Model name: **mMICE-P-130m**.

## Things I have NOT verified: check them in step 0 and tell me what you find
1. Whether `embeddings-fine-tuning-filtered-en` has the same seven translated splits as the French one (I only read the FR dataset page).
2. The contents of `lightonai/nanobeir-multilingual` (languages, subsets, whether the English exclusions of `nanobeir11` apply).
3. Whether MIRACL has a German dev set usable for evaluation.
4. mmBERT-small: layer count, architecture, tokenizer, and whether it loads in `mice_v2_scorer` / `ModernBertMiceScorer` without changes. This decides the `n_contextualization_layers` / `n_interaction_layers` split for E5 (the 150m MICE-P uses 12 + 4).
5. The licenses of the datasets.
6. **`n_bits`**: `_common_eval.yaml` sets `plaid.n_bits: 2`, but the paper uses 4 for MICE-P. Look at the configs under `experiments/micep/evals/` and **ask me which value to use**; do not choose yourself.

## Step 0: read-only investigation (no changes)
1. Compare `multilingual-mice` with `main` and summarise what is already different.
2. Read `build_trainer` in `src/training_utils.py` (the `MSE_mixedbread_large` / `pearson_mixedbread_large` branch, ~line 184-360) and `build_ettin_reranker_v1_data` (~line 366). Find the sources of `PointwiseDistillationTrainer`, `PointwiseDistillationSampler`, `PointwiseMSELoss` and `FlattenAndShuffleDataset` in the installed `xpmir` / `datamaestro_ir` packages.
3. Find out (a) how the trainer cuts batches from the sampler's sample stream, and (b) how the learner (`xpm_torch_Learner`, Fabric) splits `batch_size` across GPUs. This decides how to implement single-split batches (the sampler would need to emit runs of `batch_size` consecutive samples from the same split, consistently with device sharding).
4. Find how evaluation collections are defined (`src/Evaluation/experiment.py`, `src/Evaluation/LocalEval`) and whether MIRACL, MLDR and multilingual NanoBEIR already exist in `datamaestro_ir`.
5. Report the answers to the "not verified" list above.

## Step 1: training data
1. Define the datasets for the six languages (translated splits only, `miracl_*` / `mldr_*` excluded).
2. Write an adapter from the dataset layout (`scores` with `document_ids` / `rerank_scores`, plus `queries` and `documents`) to the pointwise (query, document, teacher score) samples that the existing path expects. The first document id in each row is the positive, followed by 10 hard negatives.
3. Keep per-split identity (source and language) available to the sampler.

## Step 2: configuration surface (`src/configuration.py`)
- Next to `ettin_subset_exclude` (~line 438), add a language selector (default: the six languages) and an `alpha` parameter (default 1.0). Keep `ettin_subset_exclude` as is.
- Add a new `Losses`/trainer option for this data. Do not modify the existing `MSE_mixedbread_large` branch behaviour.

## Step 3: LightOn-style batch sampler
- Implement a sampler that picks one split per batch with probability proportional to `len(split) ** alpha` and fills the whole batch from that split. Restart exhausted splits with a reseeded permutation.
- Log the share of batches per language during training, so a skew is visible.
- Keep the MICE-P batch sizes (192 for phase 1, 256 for phase 2). There is no divisibility constraint.

## Step 4: experiment configs (new directory `experiments/multilingual/`, inheriting from the MICE-P configs)
Keep every value as close as possible to the MICE-P configs. Ignore comments in the YAML files, only the values matter.
- **E0**: evaluation only. SPLADE-v3 first-stage recall@100 and nDCG@10 on MIRACL (EN, ES, FR, DE) and MLDR (EN, ES, FR, DE, IT, PT).
- **E4**: `base: jhu-clsp/mmBERT-small`, cross-encoder, pointwise MSE on the multilingual data. Take the training settings from `_common.yaml` / `_ettin68m.yaml` (optimisation, loss, validation interval, etc.), with validation on multilingual NanoBEIR. Evaluate on MIRACL and MLDR (languages above) with the E0 candidates.
- **E5 phase 1**: mirror `micev2_ettin_150m_l12+4_v3.yaml`: `parent: ./_common_mice_v3.yaml`, `base` = the E4 checkpoint, `batch_size: 192`, `lr` grid `1.5e-5` / `5e-5`, `max_epochs: 5000`, `num_warmup_steps: 20000`, `nb_repetitions: 2`, `export_trained_models: true`, prompts `"[Q] "` / `"[D] "`, `freeze_LN: true`. Choose the layer split once item 4 above is answered.
- **E5 phase 2**: mirror `micev2_ettin_150m_l12+4_btnk_128.yaml`: `from_experiment` = the phase 1 experiment id, `doc_bottleneck_dim: 128`, `freeze_bottom_layers_phase1: true`, `freeze_top_layers: false`, `batch_size: 256`, same lr grid, `eval_checkpoint: [last, best]`. Evaluation uses SPLADE-v3 as first stage and FastPLAID with the `n_bits` value I confirm.
- Use the name **mMICE-P-130m** in ids/titles.

## Step 5: tests and dry runs
- Add tests under `tests/` for the language selector and the sampler: every batch comes from exactly one split, and split frequencies over many batches follow `size ** alpha`.
- Run `--run-mode DRY_RUN` on every new config.

## Step 6: review gate
Show me the full diff and a short summary of what changed per file, then wait. No commit, no push.

---

# Progress log (status on 2026-11-16)

**Restart point: step 5, dry runs.** Steps 1 to 4 are committed. Step 5 is partly done: the tests exist and pass, one dry run passed, one was cut off, three were not run.

## Commits on `multilingual-mice`
| Commit | Content | On `github` remote |
|---|---|---|
| `10d026c` | step 1, training splits | yes |
| `ea115b7` | step 2, configuration surface | yes |
| `9c2ea27` | step 3, single-split batch sampler | yes |
| `2032181` | `meanmaxsim` default for `MSE_lighton_multilingual` on multi-vector scorers | **no** |
| `a6d30a6` | step 4, experiment configs | **no** |

The last two are local only: `git push github multilingual-mice` fails in the agent session (no GitHub credentials). It has to be run from a terminal with a working login.

## Decisions taken since the brief was written
- `n_bits`: start with 4.
- Layer split for mMICE-P-130m: 12 + 4 (mmBERT-small has 22 layers, like ettin-150m).
- Evaluation is limited to **MIRACL only** for the moment (EN, ES, FR, DE, dev). MLDR (test split) is registered in `src/tests.py` as `mldr` but no config uses it.
- Validation: new option `nanobeir11_multilingual`, NanoBEIR11 in the six languages (66 sets).
- Reranker `max_epochs`: 6000 (effective `_ettin68m` value).
- Exhausted splits: `replay` is the default; one ablation config compares it with `reshuffle`, on the reranker.
- Config names carry no `eX_` prefix.
- Phase 1 and phase 2 `base`: explicit path, currently a `TODO/path/...` placeholder. It must be replaced with the folder of the model exported by `train_mmbert_small_reranker` (`<workspace>/experiments/train_mmbert_small_reranker/<run>/results/models/<name>`). Pushing the reranker to a private Hub repo was discussed and postponed.

## Configs (`experiments/multilingual/`)
| File | Experiment id | Dry run |
|---|---|---|
| `splade_first_stage.yaml` | `eval_splade_first_stage_miracl` | passed |
| `mmbert_small_reranker.yaml` | `train_mmbert_small_reranker` | cut off by a 50 min limit while loading the last LightOn split; trainer not planned yet |
| `mmbert_small_reranker_abl_exhausted_split.yaml` | `train_mmbert_small_reranker_abl_exhausted_split` | not run |
| `mmicep_130m_l12+4.yaml` | `train_mmicep_130m_l12+4` | not run, needs the real `base` path |
| `mmicep_130m_l12+4_btnk_128.yaml` | `train_mmicep_130m_l12+4_plaid_128` | not run, needs the real `base` path and the phase 1 experiment |

## How the dry runs were launched on the lab machine (`zz`)
Use the separate environment `~/venvs/mice-multilingual`. Two local obstacles needed scratch files, which may have been deleted since (the scratchpad is not durable); recreate them if so.

1. `~/.config/experimaestro/launchers.py` rejects the cluster-sized requirements of the MICE-P configs. A temporary config directory was used, with:
   - `launchers.py`: `find_launcher(requirements, tags=set())` returning `DirectLauncher(connector=LocalConnector.instance())`
   - `settings.yaml`: one workspace pointing to a scratch directory
2. The MIRACL document stores are not built on this machine, and the training experiments open every evaluation corpus at planning time. An override file was used:
   ```yaml
   evaluation:
       datasets:
           - nano_nq
   preprocessing:
       blocking_download: true
   ```

Command (with `$S` the directory holding those files):
```
HF_HOME=/home/vast/.cache/huggingface \
~/venvs/mice-multilingual/bin/experimaestro run-experiment --run-mode dry_run \
  --xpm-config-dir $S/xpmcfg --workdir $S/xpm-dryrun \
  --pre-yaml $S/dry_override.yaml \
  experiments/multilingual/mmbert_small_reranker.yaml
```
`splade_first_stage.yaml` needs neither the override file nor MIRACL data. Lists cannot be overridden with `-c` (they arrive as strings); use `--pre-yaml`.

## Known issues to handle before the real runs
- **Set `HF_HOME` on `zz`** (found 2026-11-17). The machine sets `XDG_CACHE_HOME=/local/vast/.cache` system-wide, so Hugging Face caches to `/local`, which sits on the root disk (`/dev/md0`, full: under 100 MB free, filled by other users). Downloads then fail with `No space left on device` / `Not enough disk space`, surfacing as `DownloadError` for a LightOn split. Export `HF_HOME=/home/vast/.cache/huggingface` (NFS, where the 28 GB of LightOn splits already are) for every dry run and real job on `zz`. `TMPDIR` does not matter.
- **MIRACL must be prepared first.** Without the four MIRACL document stores, the training configs fail at planning with `OSError: No such file or directory`. Prepare them beforehand or set `preprocessing.blocking_download: true`.
- **Planning loads all the training data.** Launching a training config makes the experiment manager download and generate the 42 LightOn splits (about 28 GB in `~/.cache/huggingface/datasets`, around 15 minutes the first time on `zz`).
- **PLAID in phase 2** indexes the four full MIRACL corpora (about 74M passages) for 8 models. The index size has not been measured.
- Not verified: that the exported reranker folder loads directly as `base` in phase 1; that the first-stage runs of `splade_first_stage.yaml` are reused by the training experiments.

## What is left
1. Step 5: finish the dry run of `mmbert_small_reranker.yaml`, then the ablation config. Phase 1 and phase 2 can only be dry-run once `base` points to a real model.
2. Step 6: review gate (full diff and per-file summary).
3. Push `2032181` and `a6d30a6`.
4. Later: re-enable MLDR in the evaluation, decide on the Hub route for the reranker.
