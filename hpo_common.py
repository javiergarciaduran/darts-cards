"""Shared helpers for the augment-phase HPO scripts (random search and TPE).

Both `hpo_random_search_baseline.py` and `hpo_tpe.py` sample augment.py
hyperparameters, launch augment.py as a subprocess for each trial, parse the
final validation accuracy from its log output, and record every trial result
to a CSV file. This module centralizes that shared logic so the two scripts
cannot drift apart on bug fixes (see bugs_fixed.md).
"""

import csv
import os
import re
import subprocess
import sys
import threading
from pathlib import Path

import optuna

optuna.logging.set_verbosity(optuna.logging.WARNING)

# Directory containing this module (and augment.py), used as the cwd for the
# augment.py subprocess so its local imports / relative paths resolve, and as
# the anchor for the Optuna SQLite study database regardless of the caller's
# current working directory.
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))

# Path to the shared Optuna SQLite study database. Anchored to SCRIPT_DIR (not
# a bare relative path) so it always lands on local disk next to this script,
# never accidentally on a Google Drive mount - sqlite over a Drive mount is
# unreliable and can corrupt the study.
STUDY_DB_PATH = Path(SCRIPT_DIR, "hpo_study.db").as_posix()
STUDY_STORAGE_URL = "sqlite:///" + STUDY_DB_PATH

# Kill an augment.py trial if it runs longer than this, even if it never
# produces output (a bare process.wait(timeout=...) after the stdout loop
# would never fire, since the loop blocks until EOF).
TRIAL_TIMEOUT_SECONDS = 1800

GENOTYPE = (
    "Genotype(normal=[[('dil_conv_3x3', 1), ('dil_conv_5x5', 0)], "
    "[('dil_conv_3x3', 1), ('dil_conv_5x5', 2)], "
    "[('dil_conv_5x5', 3), ('dil_conv_3x3', 1)], "
    "[('dil_conv_3x3', 3), ('sep_conv_3x3', 4)]], "
    "normal_concat=range(2, 6), "
    "reduce=[[('max_pool_3x3', 1), ('skip_connect', 0)], "
    "[('max_pool_3x3', 1), ('dil_conv_5x5', 2)], "
    "[('max_pool_3x3', 1), ('skip_connect', 3)], "
    "[('max_pool_3x3', 1), ('sep_conv_5x5', 3)]], "
    "reduce_concat=range(2, 6))"
)

FINAL_PREC_RE = re.compile(r"Final best Prec@1 = ([\d.]+)%")

CSV_FIELDNAMES = [
    "trial_number",
    "value",
    "lr",
    "weight_decay",
    "drop_path_prob",
    "aux_weight",
    "cutout_length",
]


def append_csv_row(csv_path, row):
    """Append a single trial result row to the results CSV, creating it with a header if needed."""
    csv_file = Path(csv_path)
    file_has_header = csv_file.exists() and csv_file.stat().st_size > 0
    with open(csv_file, "a", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=CSV_FIELDNAMES)
        if not file_has_header:
            writer.writeheader()
        writer.writerow(row)


def count_completed_csv_trials(csv_path):
    """Return the number of trials already recorded in the CSV."""
    p = Path(csv_path)
    if not p.exists():
        return 0
    with open(p, newline="") as f:
        return sum(1 for _ in csv.DictReader(f))


def trials_dataframe_path(csv_path):
    """Derive the trials-dataframe output path from the results CSV path.

    Includes the results CSV's own stem so that the random-search and TPE
    runs (which by default share the same parent directory) don't overwrite
    each other's dataframe dump.
    """
    p = Path(csv_path)
    return p.with_name(p.stem + "_trials_dataframe.csv")


def sample_hyperparameters(trial):
    """Sample the augment.py hyperparameters shared by both HPO strategies."""
    return {
        "lr": trial.suggest_float("lr", 0.005, 0.05, log=True),
        "weight_decay": trial.suggest_float("weight_decay", 1e-4, 1e-3, log=True),
        "drop_path_prob": trial.suggest_float("drop_path_prob", 0.05, 0.40),
        "aux_weight": trial.suggest_float("aux_weight", 0.2, 0.6),
        "cutout_length": trial.suggest_categorical("cutout_length", [8, 12, 16]),
    }


def _build_hyperparameter_distributions():
    """Derive Optuna distributions matching sample_hyperparameters.

    Built by probing sample_hyperparameters with a throwaway in-memory study
    so the search space used for replay (seed_study_from_csv) cannot
    silently drift out of sync with the one actually used to sample trials.
    """
    probe_study = optuna.create_study()
    probe_trial = probe_study.ask()
    sample_hyperparameters(probe_trial)
    return dict(probe_trial.distributions)


HYPERPARAMETER_DISTRIBUTIONS = _build_hyperparameter_distributions()


def seed_study_from_csv(study, csv_path):
    """Replay completed trials from a results CSV into a freshly created study.

    On Colab the SQLite study database (STUDY_DB_PATH) lives on the local,
    ephemeral disk and is lost whenever the runtime disconnects, while the
    results CSV lives on Drive and survives. If `study` comes back with no
    trials (because its backing database was just (re)created) but the CSV
    already has rows from earlier sessions, replay them as COMPLETE/PRUNED
    trials so that:
      - `trial.number` for the next trial continues from where the CSV left
        off, instead of restarting at 0 and colliding with earlier trial
        directories / CSV rows.
      - Samplers that learn from history (e.g. TPESampler) get to rebuild
        their model of the search space instead of starting from scratch
        every session.
      - The final `study.trials_dataframe()` export reflects the full run,
        not just the current session.

    Does nothing if the study already has trials (its database was not
    lost) or if the CSV does not exist yet.
    """
    if len(study.trials) > 0:
        return

    csv_file = Path(csv_path)
    if not csv_file.exists():
        return

    with open(csv_file, newline="") as f:
        rows = list(csv.DictReader(f))

    for i, row in enumerate(rows):
        try:
            params = {
                "lr": float(row["lr"]),
                "weight_decay": float(row["weight_decay"]),
                "drop_path_prob": float(row["drop_path_prob"]),
                "aux_weight": float(row["aux_weight"]),
                "cutout_length": int(row["cutout_length"]),
            }
            value = float(row["value"])
        except (KeyError, ValueError) as exc:
            print("WARNING: skipping malformed row {} in {}: {}".format(i, csv_path, exc))
            continue

        if value == -1.0:
            state = optuna.trial.TrialState.PRUNED
            value = None
        else:
            state = optuna.trial.TrialState.COMPLETE

        try:
            study.add_trial(optuna.trial.create_trial(
                state=state,
                value=value,
                params=params,
                distributions=HYPERPARAMETER_DISTRIBUTIONS,
            ))
        except ValueError as exc:
            print("WARNING: could not replay row {} from {}: {}".format(i, csv_path, exc))


def run_augment_trial(trial, params, trial_name, hpo_output_dir, data_path="./data/cards"):
    """Launch augment.py for one trial and return its Final Prec@1 (0-1) or -1.0 on failure.

    Writes the subprocess output to <hpo_output_dir>/<trial_name>/stdout.txt
    and echoes progress lines to stdout, tagged with the trial number.
    """
    trial_dir = Path(hpo_output_dir) / trial_name
    trial_dir.mkdir(parents=True, exist_ok=True)
    stdout_path = trial_dir / "stdout.txt"

    cmd = [
        sys.executable, "-u", "augment.py",
        "--name", trial_name,
        "--dataset", "cards",
        "--data_path", data_path,
        "--path", str(hpo_output_dir),
        "--batch_size", "96",
        "--init_channels", "24",
        "--layers", "14",
        "--momentum", "0.9",
        "--grad_clip", "5.0",
        "--print_freq", "50",
        "--workers", "2",
        "--gpus", "0",
        "--seed", "42",
        "--epochs", "50",
        "--genotype", GENOTYPE,
        "--lr", str(params["lr"]),
        "--weight_decay", str(params["weight_decay"]),
        "--drop_path_prob", str(params["drop_path_prob"]),
        "--aux_weight", str(params["aux_weight"]),
        "--cutout_length", str(params["cutout_length"]),
    ]

    try:
        process = subprocess.Popen(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            cwd=SCRIPT_DIR,
        )
    except OSError as exc:
        print("[Trial {}] ERROR: failed to launch augment.py: {}".format(trial.number, exc))
        return -1.0, -1

    timeout_event = threading.Event()

    def _kill_on_timeout():
        timeout_event.set()
        try:
            process.kill()
        except OSError:
            pass

    timer = threading.Timer(TRIAL_TIMEOUT_SECONDS, _kill_on_timeout)
    timer.start()
    try:
        try:
            with open(stdout_path, "w") as log_file:
                for line in process.stdout:
                    log_file.write(line)
                    if "Prec@1" in line or "LR" in line or "Logger" in line:
                        print("[Trial {}] {}".format(trial.number, line.rstrip()))
            return_code = process.wait()
        except Exception as exc:
            print("[Trial {}] ERROR while running augment.py: {}".format(trial.number, exc))
            process.kill()
            process.wait()
            return -1.0, -1
    finally:
        timer.cancel()

    if timeout_event.is_set():
        print("[Trial {}] WARNING: trial exceeded timeout and was killed.".format(trial.number))
        # Don't force return_code = -1 here: process.kill() already makes
        # process.wait() return a non-zero/negative code on both POSIX and
        # Windows when the kill actually happened. Overwriting a genuine 0
        # would discard a trial that finished successfully right as the
        # timer fired.

    value = -1.0
    if return_code == 0:
        with open(stdout_path, "r") as log_file:
            for line in log_file:
                match = FINAL_PREC_RE.search(line)
                if match:
                    value = float(match.group(1)) / 100.0
                    break
        if value == -1.0:
            print("[Trial {}] WARNING: could not parse Final best Prec@1 from log".format(trial.number))
    else:
        print("[Trial {}] WARNING: augment.py exited with code {}".format(trial.number, return_code))

    return value, return_code


def make_objective(args, trial_name_prefix):
    """Build the Optuna objective function shared by the random-search and TPE scripts.

    `trial_name_prefix` distinguishes the two scripts' trial directories /
    log files (e.g. "hpo_rs_trial_" vs "hpo_tpe_trial_") under args.hpo_output_dir.
    """

    def objective(trial):
        """Sample hyperparameters, run augment.py as a subprocess, and return Final Prec@1."""
        params = sample_hyperparameters(trial)
        trial_name = "{}{:02d}".format(trial_name_prefix, trial.number)

        value, return_code = run_augment_trial(trial, params, trial_name, args.hpo_output_dir, args.data_path)

        try:
            append_csv_row(args.csv_path, {
                "trial_number": trial.number,
                "value": value,
                "lr": params["lr"],
                "weight_decay": params["weight_decay"],
                "drop_path_prob": params["drop_path_prob"],
                "aux_weight": params["aux_weight"],
                "cutout_length": params["cutout_length"],
            })
        except OSError as exc:
            print("[Trial {}] WARNING: failed to append result to {}: {}".format(
                trial.number, args.csv_path, exc))

        if return_code != 0 or value == -1.0:
            raise optuna.exceptions.TrialPruned()

        return value

    return objective
