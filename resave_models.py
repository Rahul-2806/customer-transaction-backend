import pickle
import os

# Load from original location
orig_dir = r"C:\DATA SCIENCE PROJECTS\Customer Transaction Prediction\models"
save_dir = r"C:\DATA SCIENCE PROJECTS\Customer Transaction Prediction\customer-transaction-backend\models"

os.makedirs(save_dir, exist_ok=True)

print("Testing original files...")
for fname in ["lgb_models.pkl", "xgb_models.pkl", "cat_models.pkl"]:
    path = os.path.join(orig_dir, fname)
    try:
        with open(path, "rb") as f:
            data = pickle.load(f)
        print(f"✅ {fname} loads OK — type: {type(data)}, len: {len(data)}")
        
        # Re-save with protocol 4
        save_path = os.path.join(save_dir, fname)
        with open(save_path, "wb") as f:
            pickle.dump(data, f, protocol=4)
        print(f"✅ Re-saved {fname} to backend/models/")
    except Exception as e:
        print(f"❌ {fname} failed: {e}")

# Copy config
import shutil
shutil.copy(
    os.path.join(orig_dir, "ensemble_config.json"),
    os.path.join(save_dir, "ensemble_config.json")
)
print("✅ ensemble_config.json copied")
print("\n✅ All done!")