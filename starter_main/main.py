# Put the code for your API here.
import json
from typing import List
import os

import numpy as np
import pandas as pd
from fastapi import FastAPI
from pydantic import BaseModel, Field

import starter_main.starter.ml.data as mldata
import starter_main.starter.ml.model as mlmodel
from starter_main.starter.train_model import silice_test_performance


class Item(BaseModel):
    age: float
    workclass: str
    fnlgt: float
    education: str
    education_num: float  = Field(None, alias='education-num')
    marital_status: str = Field(None, alias='marital-status')
    occupation: str
    relationship: str
    race: str
    sex: str
    capital_gain: float = Field(None, alias='capital-gain')
    capital_loss: float = Field(None, alias='capital-loss')
    hours_per_week: float = Field(None, alias='hours-per-week')
    native_country: str = Field(None, alias='native-country')

    class Config:
        allow_population_by_field_name = True
        schema_extra = {
            "example": {
                "age": 42.0,
                "workclass": "State-gov",
                "fnlgt": 77516,
                "education": "Bachelors",
                "education_num": 13.0,
                "marital_status": "Never-married",
                "occupation": "Adm-clerical",
                "relationship": "Not-in-family",
                "race": "White",
                "sex": "Male",
                "capital_gain": 2174,
                "capital_loss": 0,
                "hours_per_week": 40,
                "native-country": "United-States"

            }
        }

class Features(BaseModel):
    feature_list: List[str]


    class Config:
        allow_population_by_field_name = True
        schema_extra = {
            "example": {
                "feature_list":  [
                    "workclass",
                    "education",
                    "marital-status",
                    "occupation",
                    "relationship",
                    "race",
                    "sex",
                    "native-country",
                ]


            }
        }


app = FastAPI()


@app.get("/")
async def root():
    return {"message": "Hello World"}

@app.post("/prediction/")
async def prediction(item: Item):
    #convert pydantic to pandas dataframe
    df = pd.DataFrame([item.dict(by_alias=True)])
    #load models
    print("load model")
    loaded_model = mlmodel.load_model('model.pkl', basepath='starter_main/model')

    print("load encoder")
    encoder = mlmodel.load_model('encoder.pkl', basepath='starter_main/model')

    print("load lb")
    lb = mlmodel.load_model('lb.pkl', basepath='starter_main/model')

    cat_features = [
        "workclass",
        "education",
        "marital-status",
        "occupation",
        "relationship",
        "race",
        "sex",
        "native-country",
    ]

    print("start process test datafunction")
    print(df)
    X_test, y_test, encoder, lb = mldata.process_data(
    df, categorical_features=cat_features, label=None, training=False,
    encoder=encoder, lb=lb
    )


    preds_test = mlmodel.inference(loaded_model,X_test)
    print(preds_test)
    lists = preds_test.tolist()
    json_str = json.dumps(lists)
    return json_str


@app.post("/slices_score/")
async def slices_score(features: Features):
    #convert list to array
    feature_list = np.array(features.feature_list)
    precision, recall, fbeta =silice_test_performance(feature_list)
    results = [precision, recall, fbeta]
    print(results)
    json_str = json.dumps(results)
    return json_str

if "DYNO" in os.environ and os.path.isdir(".dvc"):
    os.system("dvc config core.no_scm true")
    if os.system("dvc pull") != 0:
        exit("dvc pull failed")
    os.system("rm -r .dvc .apt/usr/lib/dvc")