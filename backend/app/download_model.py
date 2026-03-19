from huggingface_hub import snapshot_download

print("Starting download of all-MiniLM-L6-v2...")
# This will download the model to your default cache directory
path = snapshot_download(repo_id="sentence-transformers/all-MiniLM-L6-v2")
print(f"Model downloaded successfully to: {path}")