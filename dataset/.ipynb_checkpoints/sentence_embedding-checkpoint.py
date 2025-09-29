import json
import pickle
from sentence_transformers import SentenceTransformer
import torch

# 初始化模型
device = "cuda" if torch.cuda.is_available() else "cpu"
model = SentenceTransformer("Alibaba-NLP/gte-Qwen2-1.5B-instruct", trust_remote_code=True, device=device)

# 載入 JSON 資料庫
with open("filtered_file.json", "r", encoding="utf-8") as f:
    data = json.load(f)

# 抽出所有句子
sentences = [entry["text"] for entry in data]

# 抽出所有 animation path
animation_path = [entry["sign_animation"] for entry in data]

# 向量化全部句子（建議用 batch 處理，sentence-transformers 已自動處理）
print("Encoding database sentences...")
embeddings = model.encode(sentences, batch_size=16, convert_to_tensor=True, show_progress_bar=True)

# 儲存向量與原始資料對應關係
with open("sign_vectors.pkl", "wb") as f:
    pickle.dump((sentences, embeddings, animation_path), f)

print("資料庫向量化完成並儲存！")
