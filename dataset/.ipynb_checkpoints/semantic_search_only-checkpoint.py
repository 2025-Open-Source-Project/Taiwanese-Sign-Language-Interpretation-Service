import torchvision
torchvision.disable_beta_transforms_warning()
from sentence_transformers import util, SentenceTransformer
import pickle
import gc
import torch

# 初始化模型
device = "cuda" if torch.cuda.is_available() else "cpu"
model = SentenceTransformer("Alibaba-NLP/gte-Qwen2-1.5B-instruct", trust_remote_code=True, device=device)

# 載入資料與向量
with open("sign_vectors.pkl", "rb") as f:
    data, embeddings, animation_datapath = pickle.load(f)

# 使用者輸入句子
query = "趕快過馬路"

# 向量化輸入句子
query_embedding = model.encode(query, convert_to_tensor=True)

# 計算餘弦相似度
cos_scores = util.cos_sim(query_embedding, embeddings)[0]  # 取第一列（因為只有一個 query）

# 取前五名
top_results = torch.topk(cos_scores, k=5)

print("查詢結果：")
for score, idx in zip(top_results.values, top_results.indices):
    sentence = data[idx]               # 是一句句子，string
    animation = animation_datapath[idx]  # 是一個 path，string
    print(f"- 相似度: {score.item():.4f}｜句子: {sentence}｜動畫檔: {animation}")
