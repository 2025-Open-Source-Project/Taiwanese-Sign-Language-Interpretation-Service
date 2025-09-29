import pickle
import torch
import torchvision
torchvision.disable_beta_transforms_warning()
from huggingface_hub import snapshot_download
from sentence_transformers import SentenceTransformer

device = "cuda" if torch.cuda.is_available() else "cpu"

mistral_local_path = snapshot_download("TheBloke/Mistral-7B-Instruct-v0.2-GPTQ", local_dir="models/mistral", local_files_only=True)
# model_qwen = SentenceTransformer(local_path, device=device, local_files_only=True)
print("Store success!")