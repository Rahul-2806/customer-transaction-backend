import joblib
import numpy as np

xgb_models = joblib.load(r"C:\DATA SCIENCE PROJECTS\Customer Transaction Prediction\customer-transaction-backend\models\xgb_models.pkl")

print(f"Type: {type(xgb_models)}")
print(f"Length: {len(xgb_models)}")
print(f"First model type: {type(xgb_models[0])}")

# Try prediction
import xgboost as xgb
X = np.random.normal(0, 1, (1, 600))

model = xgb_models[0]
print(f"\nModel class: {model.__class__.__name__}")

# Try sklearn API first
try:
    pred = model.predict_proba(X)
    print(f"✅ predict_proba works: {pred}")
except Exception as e:
    print(f"predict_proba failed: {e}")

# Try booster API
try:
    dm = xgb.DMatrix(X)
    pred = model.predict(dm)
    print(f"✅ DMatrix predict works: {pred}")
except Exception as e:
    print(f"DMatrix predict failed: {e}")

# Try direct numpy
try:
    pred = model.predict(X)
    print(f"✅ Direct predict works: {pred}")
except Exception as e:
    print(f"Direct predict failed: {e}")
