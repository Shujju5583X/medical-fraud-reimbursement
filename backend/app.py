from fastapi import FastAPI, UploadFile, File, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.templating import Jinja2Templates
from pydantic import BaseModel, Field
import pandas as pd
import joblib
import numpy as np
import io
import os
from typing import List, Dict, Any
from src.feature_service import compute_features

# Initialize FastAPI app
app = FastAPI(title="Medical Fraud & Reimbursement API", version="2.0")

# Enable CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Allow all origins (update in production)
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Jinja2 template setup
templates = Jinja2Templates(directory="../frontend")  # Assuming frontend is outside backend folder

# Load model artifacts on startup
artifacts = None

@app.on_event("startup")
def load_model():
    global artifacts
    model_path = os.path.join("models", "artifacts.joblib")
    if not os.path.exists(model_path):
        raise RuntimeError(
            f"Model not found at {model_path}. Train it first using:\n"
            f'python -m src.train --data "..\\data\\persons_1000.xlsx" --out "models"'
        )
    artifacts = joblib.load(model_path)

# Serve frontend
@app.get("/", response_class=HTMLResponse)
async def serve_frontend(request: Request):
    return templates.TemplateResponse("index.html", {"request": request})

# Pydantic model for input
class PredictIn(BaseModel):
    person_id: str | None = None
    first_name: str | None = None
    last_name: str | None = None
    date_of_birth: str | None = None
    age: float | int | None = None
    email: str | None = None
    phone: str | None = None
    policy_no: str | None = None
    policy_type: str | None = None
    policy_start_date: str | None = None
    policy_end_date: str | None = None
    claim_count: float | int | None = None
    last_claim_date: str | None = None
    smoker: str | bool | int | None = None
    claim_date: str = Field(default_factory=lambda: pd.Timestamp.today().normalize().strftime("%Y-%m-%d"))

# Decision logic
def decide(row_proba: float, fraud_score: float, rules_hits: List[str]) -> Dict[str, Any]:
    hard = [h for h in rules_hits if "(HARD)" in h or "CLAIM_" in h or "MISSING_POLICY_NO" in h or "AGE_DOB_MISMATCH" in h]
    soft = [h for h in rules_hits if h not in hard]

    if len(hard) > 0:
        return {"decision": "Decline", "reason": "Hard rule violation", "rules_hits": rules_hits}

    T_LOW, T_HIGH = 0.20, 0.60
    if fraud_score < T_LOW and row_proba >= 0.50:
        return {"decision": "Approve", "reason": "Low fraud score & likely eligible", "rules_hits": rules_hits}
    if fraud_score >= T_HIGH:
        return {"decision": "Decline", "reason": "High fraud score", "rules_hits": rules_hits}
    return {"decision": "Review", "reason": "Ambiguous risk", "rules_hits": rules_hits + soft}

# Single prediction
@app.post("/predict", response_class=JSONResponse)
async def predict(item: PredictIn):
    df = pd.DataFrame([item.dict()])
    X, rules_hits = compute_features(df)

    feat_cols = artifacts["feature_columns"]
    for c in feat_cols:
        if c not in X.columns:
            X[c] = 0
    X = X[feat_cols]

    clf = artifacts["clf"]
    iso = artifacts["iso"]

    ready_proba = float(clf.predict_proba(X)[:, 1][0])
    scores = -iso.score_samples(X)
    mn, mx = float(scores.min()), float(scores.max())
    fraud_score = float(0.0 if mx == mn else (scores[0] - mn) / (mx - mn))

    decision = decide(ready_proba, fraud_score, rules_hits[0])
    return {
        "decision": decision["decision"],
        "reason": decision["reason"],
        "ready_probability": round(ready_proba, 3),
        "fraud_score": round(fraud_score, 3),
        "rules_hits": decision["rules_hits"],
        "explanations": [
            "age_vs_dob_delta and is_policy_active strongly affect risk",
            f"claim_count_rate={float(X.get('claim_count_rate', pd.Series([0])).iloc[0]):.2f}"
        ]
    }

# Bulk prediction
@app.post("/predict-bulk", response_class=JSONResponse)
async def predict_bulk(file: UploadFile = File(...)):
    content = await file.read()
    if file.filename.lower().endswith((".xlsx", ".xls")):
        df = pd.read_excel(io.BytesIO(content))
    else:
        df = pd.read_csv(io.BytesIO(content))

    if "claim_date" not in df.columns:
        df["claim_date"] = pd.Timestamp.today().normalize()

    X, rules_list = compute_features(df)

    feat_cols = artifacts["feature_columns"]
    for c in feat_cols:
        if c not in X.columns:
            X[c] = 0
    X = X[feat_cols]

    clf = artifacts["clf"]
    iso = artifacts["iso"]
    ready_proba = clf.predict_proba(X)[:, 1]

    scores = -iso.score_samples(X)
    mn, mx = float(scores.min()), float(scores.max())
    fraud = np.zeros_like(scores) if mx == mn else (scores - mn) / (mx - mn)

    out = []
    for i in range(len(df)):
        dec = decide(float(ready_proba[i]), float(fraud[i]), rules_list[i])
        out.append({
            "decision": dec["decision"],
            "reason": dec["reason"],
            "ready_probability": round(float(ready_proba[i]), 3),
            "fraud_score": round(float(fraud[i]), 3),
            "rules_hits": dec["rules_hits"]
        })

    return {"rows": len(df), "predictions": out}
