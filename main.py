import os, json, pickle, time
import joblib
import numpy as np
import pandas as pd
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import List

app = FastAPI(title="Customer Transaction Prediction API", version="1.0.0")
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"])

# Models can be in local ./models/ folder OR downloaded from HF
MODELS_DIR = os.getenv("MODELS_DIR", "models")

lgb_models     = None
xgb_models     = None
cat_models     = None
ensemble_config = None
models_loaded  = False

def load_models():
    global lgb_models, xgb_models, cat_models, ensemble_config, models_loaded
    try:
        print(f"📂 Loading models from {MODELS_DIR}...")
        lgb_models = joblib.load(f"{MODELS_DIR}/lgb_models.pkl")
        print("✅ LightGBM loaded")
        xgb_models = joblib.load(f"{MODELS_DIR}/xgb_models.pkl")
        print("✅ XGBoost loaded")
        cat_models = joblib.load(f"{MODELS_DIR}/cat_models.pkl")
        print("✅ CatBoost loaded")
        with open(f"{MODELS_DIR}/ensemble_config.json", "r") as f: ensemble_config = json.load(f)
        print("✅ Config loaded")
        models_loaded = True
        print(f"✅ All models loaded! AUC: {ensemble_config['final_auc']:.4f}")
    except Exception as e:
        print(f"❌ Model loading error: {e}")
        models_loaded = False

@app.on_event("startup")
async def startup():
    load_models()

def engineer_features(df: pd.DataFrame) -> pd.DataFrame:
    orig_cols = [f"var_{i}" for i in range(200)]
    for col in orig_cols:
        val_counts = df[col].value_counts()
        df[f"{col}_count"]  = df[col].map(val_counts).fillna(0)
        df[f"{col}_unique"] = df[col].map(df[col].value_counts()).fillna(0)
    return df

def predict_ensemble(df: pd.DataFrame) -> np.ndarray:
    feat_cols = ensemble_config["feature_cols"]
    w         = ensemble_config["final_weights"]
    X         = df[feat_cols].values

    lgb_preds = np.zeros(len(X))
    for model in lgb_models:
        lgb_preds += model.predict(X) / len(lgb_models)

    xgb_preds = np.zeros(len(X))
    for model in xgb_models:
        xgb_preds += model.predict_proba(X)[:, 1] / len(xgb_models)

    cat_preds = np.zeros(len(X))
    for model in cat_models:
        cat_preds += model.predict_proba(X)[:, 1] / len(cat_models)

    return w[0]*lgb_preds + w[1]*xgb_preds + w[2]*cat_preds

class PredictRequest(BaseModel):
    features: List[float]

class BatchPredictRequest(BaseModel):
    samples: List[List[float]]

@app.get("/")
def root():
    return {"status":"online","model":"Customer Transaction Prediction","version":"1.0.0",
            "auc": ensemble_config["final_auc"] if ensemble_config else None,"models_loaded":models_loaded}

@app.get("/health")
def health():
    return {"status":"healthy","models_loaded":models_loaded}

@app.get("/model-info")
def model_info():
    if not models_loaded: raise HTTPException(status_code=503, detail="Models not loaded")
    return {
        "model_type":          "Weighted Ensemble (LightGBM + XGBoost + CatBoost)",
        "auc":                 ensemble_config["final_auc"],
        "weights":             {"LightGBM":round(ensemble_config["final_weights"][0],4),"XGBoost":round(ensemble_config["final_weights"][1],4),"CatBoost":round(ensemble_config["final_weights"][2],4)},
        "n_folds":             ensemble_config["n_folds"],
        "n_features":          len(ensemble_config["original_feature_cols"]),
        "engineered_features": len(ensemble_config["feature_cols"]),
        "best_threshold":      ensemble_config["best_threshold"],
        "training_samples":    200000,
        "dataset":             "Santander Customer Transaction Prediction"
    }

@app.post("/predict")
def predict(req: PredictRequest):
    if not models_loaded: raise HTTPException(status_code=503, detail="Models not loaded")
    if len(req.features) != 200: raise HTTPException(status_code=400, detail=f"Expected 200 features, got {len(req.features)}")
    try:
        start     = time.time()
        orig_cols = [f"var_{i}" for i in range(200)]
        df        = pd.DataFrame([req.features], columns=orig_cols)
        df        = engineer_features(df)
        prob      = predict_ensemble(df)[0]
        threshold = ensemble_config["best_threshold"]
        return {
            "probability":      round(float(prob), 6),
            "prediction":       int(prob >= threshold),
            "prediction_label": "Transaction" if prob >= threshold else "No Transaction",
            "confidence":       round(float(max(prob, 1-prob)) * 100, 2),
            "threshold":        threshold,
            "elapsed_seconds":  round(time.time()-start, 3),
            "model":            "LightGBM + XGBoost + CatBoost Ensemble"
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/predict-batch")
def predict_batch(req: BatchPredictRequest):
    if not models_loaded: raise HTTPException(status_code=503, detail="Models not loaded")
    if not req.samples: raise HTTPException(status_code=400, detail="No samples provided")
    if len(req.samples) > 1000: raise HTTPException(status_code=400, detail="Max 1000 samples per batch")
    try:
        start     = time.time()
        orig_cols = [f"var_{i}" for i in range(200)]
        df        = pd.DataFrame(req.samples, columns=orig_cols)
        df        = engineer_features(df)
        probs     = predict_ensemble(df)
        threshold = ensemble_config["best_threshold"]
        results   = [{"probability":round(float(p),6),"prediction":int(p>=threshold),"prediction_label":"Transaction" if p>=threshold else "No Transaction","confidence":round(float(max(p,1-p))*100,2)} for p in probs]
        return {"results":results,"total":len(results),"transactions_detected":sum(r["prediction"] for r in results),"elapsed_seconds":round(time.time()-start,3)}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/sample-prediction")
def sample_prediction():
    if not models_loaded: raise HTTPException(status_code=503, detail="Models not loaded")
    np.random.seed(42)
    features = np.random.normal(0, 1, 200).tolist()
    return predict(PredictRequest(features=features))