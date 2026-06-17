"""Random-search HPO baseline for the augment phase of the DARTS-cards project.

Uses Optuna with a RandomSampler to sample augment.py hyperparameters,
launches augment.py as a subprocess for each trial, parses the final
validation accuracy from its log output, and records every trial result
to a CSV file on Google Drive (the SQLite study database itself stays on
local Colab disk - see hpo_common.STUDY_DB_PATH).
"""

import argparse
from pathlib import Path

import optuna

from hpo_common import (
    count_completed_csv_trials,
    make_objective,
    seed_study_from_csv,
    trials_dataframe_path,
    STUDY_STORAGE_URL,
)


def parse_args():
    """Parse command-line arguments for the HPO baseline script."""
    parser = argparse.ArgumentParser(description="Random-search HPO baseline for augment.py")
    parser.add_argument("--n_trials", type=int, default=20)
    parser.add_argument("--hpo_seed", type=int, default=42)
    parser.add_argument("--hpo_output_dir", type=str, default="/content/hpo_random_search")
    parser.add_argument("--csv_path", type=str, default="/content/drive/MyDrive/hpo_results.csv")
    parser.add_argument("--data_path", type=str, default="./data/cards")
    return parser.parse_args()


def main():
    """Create/resume the Optuna study and run the random-search HPO loop."""
    args = parse_args()

    Path(args.hpo_output_dir).mkdir(parents=True, exist_ok=True)
    Path(args.csv_path).parent.mkdir(parents=True, exist_ok=True)

    already_done = count_completed_csv_trials(args.csv_path)
    remaining = max(0, args.n_trials - already_done)
    if remaining == 0:
        print("All {} trials already completed. Exiting.".format(args.n_trials))
        return
    print("Resuming: {} trials done, {} remaining.".format(already_done, remaining))

    # Offset the seed by the number of trials already done so that resuming
    # after a Colab disconnect continues the random sequence instead of
    # replaying it from scratch (which would resample the same hyperparameter
    # configurations as the earlier trials).
    sampler = optuna.samplers.RandomSampler(seed=args.hpo_seed + already_done)
    study = optuna.create_study(
        direction="maximize",
        storage=STUDY_STORAGE_URL,
        study_name="random_search_baseline",
        load_if_exists=True,
        sampler=sampler,
    )
    seed_study_from_csv(study, args.csv_path)

    study.optimize(make_objective(args, "hpo_rs_trial_"), n_trials=remaining)

    try:
        print("Best trial value: {}".format(study.best_trial.value))
        print("Best trial params: {}".format(study.best_trial.params))
    except ValueError:
        print("No completed trials yet (all trials failed or were pruned).")

    df = study.trials_dataframe()
    results_path = trials_dataframe_path(args.csv_path)
    df.to_csv(results_path, index=False)
    print("Saved trials dataframe to {}".format(results_path))


if __name__ == "__main__":
    main()
