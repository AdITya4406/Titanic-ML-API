
from fastapi import FastAPI
from pydantic import BaseModel
import pandas as pd
import joblib

app = FastAPI(title="Titanic Survival Prediction API")

model = joblib.load("model.joblib")


class Passenger(BaseModel):
    Pclass: int
    Sex: str
    Age: float
    SibSp: int
    Parch: int
    Fare: float
    Embarked: str


@app.get("/")
def home():
    return {"message": "Titanic ML API is running"}


@app.post("/predict")
def predict(passenger: Passenger):
    data = pd.DataFrame([passenger.model_dump()])
    prediction = int(model.predict(data)[0])

    return {
        "survived": prediction,
        "result": "Survived" if prediction == 1 else "Did not survive"
    }
