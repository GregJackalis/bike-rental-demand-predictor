"""Run a prediction with the saved pipeline using the dummy observation.

Run from anywhere with:
    .venv/bin/python tests/test1.py
"""

from importer import load_pipeline, predict

observation1 = {
    "season": 2,
    "yr": 0,
    "mnth": 12,
    "hr": 20,             
    "holiday": 0,
    "weekday": 1,
    "workingday": 1,
    "weathersit": 1,
    "temp": 0.70,
    "atemp": 0.65,
    "hum": 0.50,
    "windspeed": 0.10,
}


model = load_pipeline()  # joblib.load; preprocessing included
predictions = predict(model=model, records=observation1, as_int=True)
print(f"Predicted bike rentals: {predictions[0]}")
