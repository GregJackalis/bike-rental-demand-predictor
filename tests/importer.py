import sys
from pathlib import Path

# `python tests/test1.py` puts tests/ (not the project root) on sys.path, so
# add the root to make `src` and `tests.obs1` importable.
PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from src.bike_rental_model import load_pipeline, predict

