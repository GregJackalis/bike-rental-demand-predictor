"""Train and save the bike rental pipeline from the command line.

    python src/train_model.py
    python src/train_model.py --n-estimators 50 --max-depth 25 --output models/lighter.joblib
    python src/train_model.py --categorical ""          # no one-hot encoding
"""

from __future__ import annotations

import argparse

try:  # works both as `python src/train_model.py` and `python -m src.train_model`
    from .bike_rental_model import (
        CATEGORICAL_FEATURES,
        DEFAULT_MODEL_PATH,
        MODEL_PARAMS,
        train_and_save,
    )
except ImportError:  # pragma: no cover - script execution
    from bike_rental_model import (
        CATEGORICAL_FEATURES,
        DEFAULT_MODEL_PATH,
        MODEL_PARAMS,
        train_and_save,
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", default=str(DEFAULT_MODEL_PATH))
    parser.add_argument("--test-size", type=float, default=0.2)
    parser.add_argument(
        "--n-estimators", type=int, default=MODEL_PARAMS["n_estimators"]
    )
    parser.add_argument(
        "--max-depth", type=int, default=MODEL_PARAMS["max_depth"]
    )
    parser.add_argument(
        "--random-state", type=int, default=MODEL_PARAMS["random_state"]
    )
    parser.add_argument(
        "--categorical",
        default=",".join(CATEGORICAL_FEATURES),
        help="Comma-separated columns to one-hot encode; use '' to disable.",
    )
    args = parser.parse_args()

    categorical = [c.strip() for c in args.categorical.split(",") if c.strip()]

    _, metrics = train_and_save(
        path=args.output,
        test_size=args.test_size,
        categorical_features=categorical,
        n_estimators=args.n_estimators,
        max_depth=args.max_depth,
        random_state=args.random_state,
    )
    print(f"Holdout MAE: {metrics['mae']:.4f}")
    print(f"Holdout R²:  {metrics['r2']:.4f}")
    print(f"Saved pipeline to {args.output}")


if __name__ == "__main__":
    main()
