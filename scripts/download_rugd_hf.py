"""Скачивание зеркала RUGD с Hugging Face (оригинальная структура папок) в data/raw/extracted."""
from huggingface_hub import snapshot_download

path = snapshot_download(repo_id="WilliamBonilla62/RUGD", repo_type="dataset",
                         local_dir="data/raw/extracted", max_workers=8)
print("DONE", path)
