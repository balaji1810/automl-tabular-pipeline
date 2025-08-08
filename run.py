from __future__ import annotations

from pathlib import Path
from sklearn.metrics import r2_score
import numpy as np
import sys
from pathlib import Path

# Ensure we import from the current project's src directory
current_dir = Path(__file__).parent
sys.path.insert(0, str(current_dir / "src"))

from automl.data import Dataset
from automl.AutoMLPipeline import AutoML
import argparse

import logging

logger = logging.getLogger(__name__)

FILE = Path(__file__).absolute().resolve()
DATADIR = FILE.parent / "data"


def main(
    task: str,
    fold: int,
    output_path: Path,
    seed: int,
    datadir: Path,
    timeout: int
):
    dataset = Dataset.load(datadir=datadir, task=task, fold=fold)

    logger.info("Fitting AutoML")


    automl = AutoML(seed=seed, timeout=timeout)
        
    automl.fit(dataset.X_train, dataset.y_train)
    test_preds: np.ndarray | tuple[np.ndarray, np.ndarray] = automl.predict(dataset.X_test)

    logger.info("Writing predictions to disk")
    with output_path.open("wb") as f:
        np.save(f, test_preds)

    if dataset.y_test is not None:
        r2_test = r2_score(dataset.y_test, test_preds)
        logger.info(f"R^2 on test set: {r2_test}")
    else:
        logger.info(f"No test set for task '{task}'")



if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--task",
        type=str,
        required=True,
        help="The name of the task to run on.",
        choices=["bike_sharing_demand", "brazilian_houses", "superconductivity", "wine_quality", "yprop_4_1", "exam_dataset"]
    )
    parser.add_argument(
        "--output-path",
        type=Path,
        default=Path("data/brazilian_houses/predictions.npy"),
        help=(
            "The path to save the predictions to."
            " By default this will just save to './predictions.npy'."
        )
    )
    parser.add_argument(
        "--fold",
        type=int,
        default=1,
        help=(
            "The fold to run on."
            " You are free to also evaluate on other folds for your own analysis."
            " For the test dataset we will only provide a single fold, fold 1."
        ),
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=10,
        help=(
            "Random seed for reproducibility if you are using and randomness,"
            " i.e. torch, numpy, pandas, sklearn, etc."
        )
    )
    
    parser.add_argument(
        "--timeout",
        type=float,
        default=60.0,
        help=(
            "Set optimization timeout"
            " i.e. 60 = 60 seconds"
        )
    )

    parser.add_argument(
        "--datadir",
        type=Path,
        default=DATADIR,
        help=(
            "The directory where the datasets are stored."
            " You should be able to mostly leave this as the default."
        )
    )
    parser.add_argument(
        "--quiet",
        action="store_true",
        help="Whether to log only warnings and errors."
    )

    args = parser.parse_args()

    if not args.quiet:
        logging.basicConfig(level=logging.INFO)
    else:
        logging.basicConfig(level=logging.WARNING)

    logger.info(
        f"Running task {args.task}"
        f"\n{args}"
    )

    main(
        task=args.task,
        fold=args.fold,
        output_path=args.output_path,
        datadir=args.datadir,
        seed=args.seed,
        timeout=args.timeout
    )