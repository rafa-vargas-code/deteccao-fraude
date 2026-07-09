from fastapi import FastAPI
from pydantic import BaseModel
import joblib
import pandas as pd

app = FastAPI(title="API de Detecção de Fraude")

modelo = joblib.load("models/modelo_xgb.joblib")
THRESHOLD = 0.7

class Transacao(BaseModel):
    Time: float
    V1: float; V2: float; V3: float; V4: float; V5: float; V6: float; V7: float
    V8: float; V9: float; V10: float; V11: float; V12: float; V13: float; V14: float
    V15: float; V16: float; V17: float; V18: float; V19: float; V20: float; V21: float
    V22: float; V23: float; V24: float; V25: float; V26: float; V27: float; V28: float
    Amount: float

@app.post("/predict")
def predict(transacao: Transacao):
    df = pd.DataFrame([transacao.model_dump()])
    prob = float(modelo.predict_proba(df)[0, 1])
    return {"probabilidade": prob, "fraude": bool(prob > THRESHOLD)}
