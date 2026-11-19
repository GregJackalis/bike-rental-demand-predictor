"""Make predictions with the saved pipeline from the command line.

    python src/predict.py --json '{"season":1,"yr":1,"mnth":6,"hr":17,
        "holiday":0,"weekday":1,"workingday":1,"weathersit":1,
        "temp":0.7,"atemp":0.65,"hum":0.5,"windspeed":0.1}'

    python src/predict.py --csv data/new_observations.csv --as-int
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

try:  # works both as `python src/predict.py` and `python -m src.predict`
    from .bike_rental_model import DEFAULT_MODEL_PATH, predict
except ImportError:  # pragma: no cover - script execution
    from bike_rental_model import DEFAULT_MODEL_PATH, predict


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", default=str(DEFAULT_MODEL_PATH))
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument("--json", help="One JSON object or a JSON list of objects")
    source.add_argument("--csv", type=Path, help="CSV with the 12 feature columns")
    parser.add_argument(
        "--as-int", action="store_true", help="Round to whole, non-negative rentals"
    )
    args = parser.parse_args()

    records = json.loads(args.json) if args.json is not None else args.csv
    if args.csv is not None:
        import pandas as pd

        records = pd.read_csv(args.csv)

    predictions = predict(records, model_path=args.model, as_int=args.as_int)
    for value in predictions:
        print(int(value) if args.as_int else f"{value:.2f}")


if __name__ == "__main__":
    main()
