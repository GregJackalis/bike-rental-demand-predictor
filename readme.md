# Bike Rental Demand Predictor

Hourly bike-rental demand regression using the UCI Bike Sharing dataset (id 275).

- `notebooks/bike_rental_demand_predictor_organised.ipynb` — analysis, experiments and the final model pipeline (section 13).
- `src/bike_rental_model.py` — reusable, notebook-free code: pipeline construction, training, saving, loading and prediction.
- `src/train_model.py` — command-line training script.
- `src/predict.py` — command-line inference script.
- `models/bike_rental_pipeline.joblib` — saved `Pipeline` (ColumnTransformer + OneHotEncoder + RandomForestRegressor).

## Why a pipeline

The notebook originally fitted a `RandomForestRegressor` directly on raw numeric columns. In section 13 the preprocessing was moved inside a `scikit-learn` `Pipeline`:

```
ColumnTransformer
  ├─ OneHotEncoder(handle_unknown="ignore")  -> holiday, weekday, workingday, weathersit
  └─ passthrough                              -> season, yr, mnth, hr, temp, atemp, hum, windspeed
RandomForestRegressor(n_estimators=100, random_state=42)
```

Because preprocessing is part of the fitted object, `joblib.load()` returns something that takes raw rows and predicts — the application never has to reproduce the encoding.

Holdout MAE / R²:

| Model | MAE | R² |
| --- | --- | --- |
| Original forest, raw codes | 44.99 | 0.9005 |
| Saved pipeline (one-hot on nominal codes) | 45.13 | 0.8999 |
| All 8 code columns one-hot encoded | 55.01 | 0.8586 |

Encoding `hr` and the other ordered codes one-hot is possible: pass a different list to `build_pipeline(categorical_features=...)` or use `--categorical` in the training script.

The "light" configuration from section 8 (`n_estimators=50`, `max_depth=25`) is a defensible alternative: in this pipeline it scores MAE 45.36 / R² 0.8983 with a 10.4 MB artifact, versus 45.13 / 0.8999 and 20.9 MB for the default. Pick it if deployment size or prediction latency matters:

```bash
.venv/bin/python src/train_model.py --n-estimators 50 --max-depth 25 --output models/bike_rental_pipeline_light.joblib
```

## Usage

From the project root, using the provided virtual environment:

```bash
.venv/bin/python src/train_model.py
.venv/bin/python src/predict.py --json '{"season":3,"yr":1,"mnth":6,"hr":17,"holiday":0,"weekday":1,"workingday":1,"weathersit":1,"temp":0.7,"atemp":0.65,"hum":0.5,"windspeed":0.1}' --as-int
.venv/bin/python src/predict.py --csv data/new_observations.csv
```

From application code:

```python
from src.bike_rental_model import load_pipeline, predict

model = load_pipeline()                       # defaults to models/bike_rental_pipeline.joblib
single = predict(model=model, records={...}, as_int=True)
many = predict(model=model, records=[{...}, {...}])
```

Input rows need these 12 columns: `season`, `yr`, `mnth`, `hr`, `holiday`, `weekday`, `workingday`, `weathersit`, `temp`, `atemp`, `hum`, `windspeed`. `cnt` is ignored if present. The calling application is responsible for deriving calendar fields from a date — the model itself does not use dates, **except** that `yr` can be derived from a `dteday` string if you include one.

`yr` is optional: if it is missing, `prepare_input()` derives it from `dteday` (2011 → 0, 2012+ → 1); if no date is given either, it uses `DEFAULT_FEATURE_VALUES["yr"] = 1`. That default means "assume the recent, higher-demand year", which is appropriate for predicting current/future demand but will overpredict historical 2011 rows. Set `DEFAULT_FEATURE_VALUES` to `{"yr": 0}` if the opposite is true for your use case.

Notes:

- Retraining downloads the dataset with `ucimlrepo` unless `data/bike_sharing.csv` exists.
- The serialized artifact was created with scikit-learn 1.9.1; use a compatible version when serving it.
- Keep the columns and their meaning consistent between training and inference. Missing or renamed columns raise a `ValueError`.
