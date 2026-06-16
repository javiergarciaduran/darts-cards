# HPO Scripts — Production Readiness Review

Scope: `hpo_random_search_baseline.py`, `hpo_tpe.py`, and the shared helper
module `hpo_common.py` (modified as part of the same fixes, since the two
target scripts delegate almost all of their logic to it).

Both scripts are designed to run repeatedly inside Colab sessions that
*will* disconnect mid-run. The repo is cloned to `/content/darts-cards`
(local, ephemeral Colab disk — see `notebooks/colab_launcher.ipynb`), while
results CSVs live on Google Drive (`/content/drive/MyDrive/...`). The
Optuna SQLite study database (`hpo_common.STUDY_DB_PATH`) is anchored next
to the scripts, i.e. on the *ephemeral* disk. This single fact — local DB
lost on every disconnect, Drive CSV survives — is the root cause of the
three High-severity issues below.

---

## hpo_common.py (shared by both scripts)

### 1. TPE re-enqueues the baseline config on every resumed session

- **Severity:** High
- **File affected:** `hpo_tpe.py` (symptom), fixed via new helper in `hpo_common.py`
- **Original problem:** `hpo_tpe.py` only enqueued the "known-good baseline"
  hyperparameters (`lr=0.025, weight_decay=3e-4, drop_path_prob=0.2,
  aux_weight=0.4, cutout_length=8` — matching the `cards_augment_v1` run
  from the notebook) when `len(study.trials) == 0`. Because the local
  SQLite study DB does not survive a Colab disconnect, `study.trials` comes
  back empty at the start of *every* session, even when the results CSV on
  Drive already shows many completed trials. The same baseline config was
  therefore re-enqueued and re-evaluated at the start of every session.
- **Implemented solution:** Added `seed_study_from_csv(study, csv_path)` in
  `hpo_common.py`. It replays every row of the results CSV into the
  freshly-created study as `COMPLETE`/`PRUNED` trials (via
  `optuna.trial.create_trial` + `study.add_trial`) *before* the enqueue
  check, so `len(study.trials) == 0` is only true on the genuinely first run
  ever. `hpo_tpe.py` now calls this helper right after `create_study`.
- **Why necessary:** Without it, every Colab reconnection burns one of the
  `remaining` trial slots on a duplicate evaluation of the exact same
  hyperparameter configuration instead of exploring new ones, and writes
  duplicate `trial_number=0` rows / `hpo_tpe_trial_00` directories into the
  persisted CSV and output dir.
- **Could it have affected experimental results:** Yes. Part of the TPE
  study's trial budget would silently be consumed by redundant repeats of
  the baseline config, and the persisted CSV would contain colliding trial
  identifiers, making downstream analysis (e.g. joining with the
  `*_trials_dataframe.csv`) unreliable.

### 2. `trial.number`-based identifiers reset across resumed sessions

- **Severity:** High
- **File affected:** both scripts, via shared `objective`/`make_objective` in `hpo_common.py`
- **Original problem:** `trial_name = "hpo_xxx_trial_{:02d}".format(trial.number)`
  and the CSV `trial_number` column both come from Optuna's in-study trial
  counter. For the same reason as issue 1 (local SQLite DB lost on
  disconnect), this counter restarts at 0 every session — even though the
  `already_done`/seed-offset logic in `main()` (and its accompanying
  comments) explicitly assumes continuity across sessions. This affects
  **both** `hpo_random_search_baseline.py` and `hpo_tpe.py` identically.
- **Implemented solution:** The same `seed_study_from_csv` replay (issue 1)
  repopulates `study.trials` with `already_done` historical trials before
  `study.optimize` runs, so the next `trial.number` naturally continues from
  `already_done` — matching the row count already in the CSV.
- **Why necessary:** `trial_number` is the join key between
  `hpo_*_results.csv` and `*_trials_dataframe.csv`, and `trial_name` is the
  on-disk directory holding each trial's log/checkpoints. Colliding values
  silently overwrite or duplicate this bookkeeping across sessions.
- **Could it have affected experimental results:** Yes. Any multi-session
  run would produce a results CSV with duplicate `trial_number`s and
  `hpo_*_trial_00` directories from different sessions, breaking any
  analysis that indexes trials by this column.

### 3. TPESampler never leaves its random "startup" phase across sessions

- **Severity:** High
- **File affected:** `hpo_tpe.py`, fixed via shared helper in `hpo_common.py`
- **Original problem:** `TPESampler(..., n_startup_trials=5, multivariate=True)`
  is instantiated fresh every session. Combined with the lost local SQLite
  DB, a fresh sampler sees `len(study.trials) == 0` at the start of each
  session. If a Colab session only completes a handful of ~50-epoch
  `augment.py` runs (very plausible) before disconnecting, the sampler may
  *never* see `n_startup_trials` (=5) completed trials within a single
  session and therefore never switches from random sampling to its TPE
  surrogate model — i.e. "TPE search" degrades to random search for its
  entire duration, defeating the purpose of comparing it against
  `hpo_random_search_baseline.py`.
- **Implemented solution:** `seed_study_from_csv` reconstructs the full
  trial history from the CSV at the start of every session, so
  `n_startup_trials` is evaluated against the *cumulative* trial count and
  the surrogate model is rebuilt from all prior results, not just the
  current session's.
- **Why necessary:** This is the central premise of having a separate TPE
  script — without persisted history, the "TPE vs random search" comparison
  is not methodologically meaningful.
- **Could it have affected experimental results:** Yes, potentially
  severely — the TPE study could silently behave like pure random search
  for its whole run, making any conclusion that "TPE found a better config
  than random search" unsupported.

### 4. Unhandled CSV-write failure aborts the entire study

- **Severity:** Medium
- **File affected:** both scripts, via shared `objective`/`make_objective` in `hpo_common.py`
- **Original problem:** `append_csv_row(args.csv_path, {...})` (writing to
  the Drive-mounted results CSV) had no error handling. `hpo_common.py`
  itself documents elsewhere that Drive-backed I/O ("sqlite over a Drive
  mount is unreliable") is a real concern in this environment. If Drive
  becomes briefly unavailable, `append_csv_row` raises `OSError`, which
  propagates out of `objective`. Since `study.optimize` was called without a
  `catch=` argument, this aborts the *entire* `study.optimize` call —
  discarding all remaining trials in the session's budget and skipping the
  final `best_trial` print and `trials_dataframe.csv` export, even though
  the just-finished `augment.py` run (potentially tens of minutes of GPU
  time) completed successfully.
- **Implemented solution:** Wrapped the `append_csv_row` call in
  `try/except OSError`, logging a warning (`"[Trial N] WARNING: failed to
  append result to <path>: <exc>"`) and continuing with the normal
  prune/return logic.
- **Why necessary:** A transient filesystem hiccup on one CSV write
  shouldn't discard an entire session's remaining trial budget and summary
  export.
- **Could it have affected experimental results:** Indirectly — if this had
  occurred previously, all results after the failure point for that session
  would have been silently lost (no `best_trial`/`trials_dataframe.csv`),
  even though prior CSV rows remained valid.

### 5. Stale documentation reference

- **Severity:** Low
- **File affected:** `hpo_common.py`
- **Original problem:** The module docstring referenced `docs/bugs_fixed.md`,
  a file that does not exist in the repository.
- **Implemented solution:** Updated the reference to `bugs_fixed.md` (this
  file, at the repository root, as requested by this review).
- **Why necessary:** Documentation accuracy / discoverability for future
  contributors following the "see X for prior bug fixes" pointer.
- **Could it have affected experimental results:** No.

---

## hpo_random_search_baseline.py

### 6. Duplicated `make_objective`/`objective` implementation

- **Severity:** Medium
- **Files affected:** `hpo_random_search_baseline.py`, `hpo_tpe.py`, `hpo_common.py`
- **Original problem:** Both scripts defined a byte-for-byte identical
  ~25-line `make_objective`/`objective` pair, differing only in the trial
  name prefix string (`"hpo_rs_trial_"` vs `"hpo_tpe_trial_"`). This directly
  contradicts `hpo_common.py`'s own stated purpose — "This module
  centralizes [shared] logic so the two scripts cannot drift apart on bug
  fixes" — since the most important piece of shared logic (running a trial,
  recording it, deciding success/prune) was *not* actually shared.
- **Implemented solution:** Moved `make_objective` into `hpo_common.py`,
  parameterized by `trial_name_prefix`. Both scripts now do:
  ```python
  study.optimize(make_objective(args, "hpo_rs_trial_"), n_trials=remaining)
  ```
  (and `"hpo_tpe_trial_"` in `hpo_tpe.py`), and import `make_objective` /
  `seed_study_from_csv` instead of `append_csv_row`, `run_augment_trial`,
  `sample_hyperparameters`, which are no longer needed directly.
- **Why necessary:** Prevents future divergence between the two scripts'
  result-recording behavior, and was a prerequisite for applying fixes 1–4
  consistently to both.
- **Could it have affected experimental results:** Not directly, but it was
  the source of the risk that future fixes (like 1–4) would be applied to
  one script and forgotten in the other.

### Integration changes for fixes 1–4

- Added `seed_study_from_csv(study, args.csv_path)` immediately after
  `optuna.create_study(...)`.
- `study.optimize(...)` now calls `make_objective(args, "hpo_rs_trial_")`.

No other issues were found specific to this file — its `parse_args`
defaults, `RandomSampler` seed-offset comment/logic, and final
`best_trial`/`trials_dataframe` reporting were all correct and consistent
with `hpo_tpe.py`.

---

## hpo_tpe.py

### Integration changes for fixes 1–4

- Added `seed_study_from_csv(study, args.csv_path)` immediately after
  `optuna.create_study(...)`, **before** the `enqueue_trial` check (this is
  what makes fix 1's `len(study.trials) == 0` check correct again).
- Added a comment explaining why the enqueue check is now safe across
  sessions.
- `study.optimize(...)` now calls `make_objective(args, "hpo_tpe_trial_")`.

Verified as *not* a bug (kept as-is): the enqueued baseline config
(`lr=0.025, weight_decay=3e-4, drop_path_prob=0.2, aux_weight=0.4,
cutout_length=8`) exactly matches the hyperparameters used for the
`cards_augment_v1` run in `notebooks/colab_launcher.ipynb` (note
`cutout_length=8` there vs. `config.py`'s bare default of `16` — the
enqueued trial intentionally reproduces the notebook's baseline run, not
`AugmentConfig`'s argparse defaults). `n_startup_trials=5` and
`consider_prior=True` (the latter is also `TPESampler`'s default, kept for
clarity) are reasonable given `n_trials=20`.

---

## Summary

- **Total issues found and fixed:** 6
  - Critical: 0
  - High: 3 (issues 1–3 — all stem from the same "local SQLite DB lost on
    Colab disconnect, Drive CSV survives" root cause)
  - Medium: 2 (issues 4, 6)
  - Low: 1 (issue 5)

### Verified correct (no changes needed)

- **Optimization direction / objective definition:** `augment.py` logs
  `"Final best Prec@1 = {:.4%}".format(best_top1)` where `best_top1` is the
  best *validation* top-1 accuracy in `[0, 1]` (via `utils.accuracy`).
  `FINAL_PREC_RE` + `/100.0` correctly recovers this `[0, 1]` value, and
  `direction="maximize"` is correct.
- **Train/val/test separation:** both HPO scripts optimize *validation*
  accuracy (`augment.py --dataset cards`, `validation=True`); the held-out
  test set is only touched once, manually, in the notebook's final
  evaluation cell — no leakage.
- **Seed-offset resumability trick** (`hpo_seed + already_done` for both
  `RandomSampler` and `TPESampler`) is sound and consistent between both
  scripts.
- **Sentinel value `-1.0`** for failed/unparseable trials cannot collide
  with a real accuracy (`[0, 1]`), so the `value == -1.0` checks are safe.
- **TPE enqueue defaults** match the notebook's documented baseline run
  (see above) — intentional, not a bug.

### Remaining concerns intentionally left unchanged

- **Single fixed `--seed 42`** for every `augment.py` trial (hardcoded in
  `hpo_common.run_augment_trial`, identical for both scripts). This means
  both HPO strategies optimize for "best config under one specific seed"
  rather than averaging over training stochasticity, which adds variance to
  the comparison. Fixing this (e.g. multi-seed evaluation per trial) would
  multiply compute cost ~N× and is a methodology decision beyond the scope
  of this review.
- **Partial CSV replay edge case:** if a row in the results CSV were
  malformed (manual edits, partial writes), `seed_study_from_csv` skips it
  with a warning rather than failing the whole run, but this would leave
  `len(study.trials) < already_done` for one trial, potentially producing
  one duplicate `trial_number`/`trial_name` in that session. In normal
  operation the CSV is only ever written by `append_csv_row` via
  `csv.DictWriter`, so malformed rows should not occur.
- **`TrialPruned` is overloaded** to mean both "the subprocess
  failed/timed out/crashed" and (if a pruner were ever wired up) "this
  config was pruned early for poor performance." Currently no pruner acts on
  intermediate values (none are reported via `trial.report`), so in practice
  `state == PRUNED` always means "infrastructure failure," but this is not
  obvious from `trials_dataframe.csv` alone. Left as-is since changing it
  would require a broader convention change (e.g. a custom `FAIL` state or
  user attribute) across both scripts.
- **Shared SQLite file (`hpo_study.db`) across both study names**
  (`random_search_baseline`, `tpe_search`) is fine for sequential use (as
  the notebook runs them), but concurrent execution of both scripts against
  the same file was not stress-tested.
