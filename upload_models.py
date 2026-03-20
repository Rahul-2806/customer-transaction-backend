from huggingface_hub import HfApi
import os

token = input("Enter your HuggingFace token: ").strip()
api = HfApi(token=token)

repo_id = "RAHULSR2806/customer-transaction-models"
models_dir = r"C:\DATA SCIENCE PROJECTS\Customer Transaction Prediction\models"

# Delete and recreate repo
print("Deleting old repo...")
try:
    api.delete_repo(repo_id=repo_id, repo_type="dataset", token=token)
    print("✅ Old repo deleted")
except:
    print("Repo not found, creating fresh...")

print("Creating new repo...")
api.create_repo(repo_id=repo_id, repo_type="dataset", token=token, private=False)
print("✅ New repo created")

# Upload files one by one using upload_file with force
files = [
    "lgb_models.pkl",
    "xgb_models.pkl",
    "cat_models.pkl",
    "ensemble_config.json"
]

for fname in files:
    path = os.path.join(models_dir, fname)
    size = os.path.getsize(path) / 1024 / 1024
    print(f"Uploading {fname} ({size:.1f} MB)...")
    with open(path, "rb") as f:
        api.upload_file(
            path_or_fileobj=f,
            path_in_repo=fname,
            repo_id=repo_id,
            repo_type="dataset",
            token=token,
        )
    print(f"✅ {fname} uploaded!")

print("\n✅ All done! Models uploaded successfully.")
