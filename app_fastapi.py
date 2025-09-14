from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
from typing import List, Literal
import joblib, pandas as pd
from pathlib import Path

MODEL_PATH = Path("outputs/model_gbr.joblib")  # run train.py first
model = joblib.load(MODEL_PATH)

class Record(BaseModel):
    age: int = Field(..., ge=0, le=120)
    sex: Literal["male","female"]
    bmi: float = Field(..., ge=10, le=80)
    children: int = Field(..., ge=0, le=12)
    smoker: Literal["yes","no"]
    region: Literal["northeast","northwest","southeast","southwest"]

class PredictRequest(BaseModel):
    records: List[Record]

app = FastAPI(title="Premium Pricing API", version="1.0")

@app.get("/health")
def health():
    return {"status": "ok"}

@app.post("/predict")
def predict(req: PredictRequest):
    if not req.records:
        raise HTTPException(status_code=400, detail="No records provided")
    df = pd.DataFrame([r.dict() for r in req.records])
    preds = model.predict(df)
    return {"n": len(preds), "predictions": [float(x) for x in preds]}
