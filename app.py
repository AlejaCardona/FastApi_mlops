# -*- coding: utf-8 -*-

import pandas as pd
from fastapi import FastAPI
import uvicorn
from pydantic import create_model
import pickle
import numpy as np
from sklearn.ensemble import GradientBoostingRegressor
# from sklearn.metrics import mean_squared_error, r2_score
# from sklearn.datasets import fetch_california_housing
# from sklearn.model_selection import train_test_split
import pandas as pd


# Crear la api
app = FastAPI()

# Load trained Pipeline
# model = load_model("best_model_Gradient_Boosting.pkl")

with open("best_model_Gradient_Boosting.pkl", 'rb') as f:
    model = pickle.load(f)

# Create input/output pydantic models
input_model = create_model("best_model_Gradient_Boosting_input", **{'time':1002,
    'trt':3,
    'age':61,
    'wtkg':494424,
    'hemo':0,
    'homo':0,
    'drugs':0,
    'karnof':90,
    'oprior':0,
    'z30':1,
    'preanti':895,
    'race':0,
    'gender':0,
    'str2':1,
    'strat':3,
    'symptom':0,
    'treat':1,
    'offtrt':0,
    'cd40':162,
    'cd420':218,
    'cd80':392,
    'cd820':564})
output_model = create_model("best_model_Gradient_Boosting_output", prediction=1)


# Define predict function
@app.post("/predict", response_model=output_model)
def predict(data: input_model):
    data = pd.DataFrame([data.dict()])
    predictions =  model.predict(data)
    # predictions = predict_model(model, data=data)
    # return {"prediction": predictions["prediction_label"].iloc[0]}
    return {"prediction": predictions}


if __name__ == "__main__":
    uvicorn.run(app, host="127.0.0.1", port=8000)
