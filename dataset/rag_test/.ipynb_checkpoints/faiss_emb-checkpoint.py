import pickle
import faiss
import torch
import numpy as np
#########

# import tensorflow.compat.v1 as tf
# tf.disable_v2_behavior()
# import os

# #choosing the 2nd  GPU card.
# os.environ["CUDA_VISIBLE_DEVICES"] = "0"

# #Occupies 20% of the selected GPU card memory.
# gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=0.2)
# sess = tf.Session(config=tf.ConfigProto(gpu_options=gpu_options))
# #########


# 讀入原始資料
with open("sign_vectors.pkl", "rb") as f:
    sentences, embeddings, animation_paths = pickle.load(f)

# 轉為 numpy 並做 L2 normalize
if isinstance(embeddings, torch.Tensor):
    embeddings = embeddings.cpu().numpy()
embeddings = embeddings.astype("float32")

# Normalize 所有向量
faiss.normalize_L2(embeddings)

# 建立 cosine 相似度用的索引
dimension = embeddings.shape[1]
index = faiss.IndexFlatIP(dimension)  # IP = Inner Product
index.add(embeddings)

# 儲存索引與 metadata
faiss.write_index(index, "sign_index_cosine.faiss")
with open("sign_metadata.pkl", "wb") as f:
    pickle.dump((sentences, animation_paths), f)

print("已建立 cosine 相似度索引並儲存！")
