from sentence_transformers import SentenceTransformer
import faiss
import pickle
import torch
import numpy as np

# 載入模型
device = "cuda" if torch.cuda.is_available() else "cpu"
model = SentenceTransformer("Alibaba-NLP/gte-Qwen2-1.5B-instruct", trust_remote_code=True, device=device)

# 載入索引 & metadata
index = faiss.read_index("sign_index_cosine.faiss")
with open("sign_metadata.pkl", "rb") as f:
    sentences, animation_paths = pickle.load(f)

# 使用者查詢
query = "這桌子少了一隻腳"

# 編碼並 normalize
query_embedding = model.encode(query, convert_to_tensor=True)
query_embedding = query_embedding.cpu().numpy().astype("float32")
if query_embedding.ndim == 1:
    query_embedding = query_embedding.reshape(1, -1)
faiss.normalize_L2(query_embedding)

# 查詢 top-k
k = 5
D, I = index.search(query_embedding.reshape(1, -1), k)

# 顯示結果（注意：值越大越接近 1，表示越相似）
print("查詢結果（Cosine Similarity）：")
for score, idx in zip(D[0], I[0]):
    print(f"- 相似度: {score:.4f}｜句子: {sentences[idx]}｜動畫: {animation_paths[idx]}")
