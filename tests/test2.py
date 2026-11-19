from importer import load_pipeline, predict

# Not yr and dteday 
observation = {
    "season": 2,
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
    # "yr" omitted
    # optionally: "dteday": "2012-12-15"
}

print(predict(records=observation, as_int=True)[0])