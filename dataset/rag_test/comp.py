import time
import pickle
import torch
import faiss
from sentence_transformers import SentenceTransformer, util
import numpy as np

# 設定
query = "Hello 早安"
top_k = 5
device = "cuda" if torch.cuda.is_available() else "cpu"

# 載入模型
model = SentenceTransformer("Alibaba-NLP/gte-Qwen2-1.5B-instruct", trust_remote_code=True, device=device)

# ============ 方法 A：原生 cosine 查詢 ============
with open("sign_vectors.pkl", "rb") as f:
    sentences, embeddings, animation_paths = pickle.load(f)

if isinstance(embeddings, torch.Tensor):
    embeddings = embeddings.to(device)
query_embedding = model.encode(query, convert_to_tensor=True).to(device)

start_a = time.time()
cos_scores = util.cos_sim(query_embedding, embeddings)[0]
top_a = torch.topk(cos_scores, k=top_k)
end_a = time.time()

# ============ 方法 B：FAISS 查詢 ============
index = faiss.read_index("sign_index_cosine.faiss")
with open("sign_metadata.pkl", "rb") as f:
    faiss_sentences, faiss_paths = pickle.load(f)

query_embedding_faiss = model.encode(query, convert_to_tensor=True).cpu().numpy().astype("float32")
query_embedding_faiss = query_embedding_faiss.reshape(1, -1)
faiss.normalize_L2(query_embedding_faiss)

start_b = time.time()
D, I = index.search(query_embedding_faiss, top_k)
end_b = time.time()

# ============ 顯示比較結果 ============
print("🟡 原生查詢耗時：{:.4f} 秒".format(end_a - start_a))
for score, idx in zip(top_a.values, top_a.indices):
    print(f"  - {score.item():.4f}｜{sentences[idx]}")

print("\n🔵 FAISS 查詢耗時：{:.4f} 秒".format(end_b - start_b))
for score, idx in zip(D[0], I[0]):
    print(f"  - {score:.4f}｜{faiss_sentences[idx]}")
