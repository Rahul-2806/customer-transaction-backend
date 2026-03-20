import pickle
import os

models_dir = "models"

for fname in ["lgb_models.pkl", "xgb_models.pkl", "cat_models.pkl"]:
    path = os.path.join(models_dir, fname)
    size = os.path.getsize(path)
    print(f"\n--- {fname} ---")
    print(f"Size: {size/1024/1024:.1f} MB")
    with open(path, "rb") as f:
        header = f.read(4)
    print(f"Header bytes: {header.hex()}")
    print(f"First byte: {hex(header[0])}")
    
    if header[0] == 0x80:
        print("✅ Valid pickle format")
        try:
            with open(path, "rb") as f:
                data = pickle.load(f)
            print(f"✅ Loaded successfully! Type: {type(data)}, Length: {len(data)}")
        except Exception as e:
            print(f"❌ pickle.load failed: {e}")
    elif header[0] == 0x02:
        print("❌ Git LFS pointer file — not actual model data")
    else:
        print(f"❌ Unknown format: {header}")
