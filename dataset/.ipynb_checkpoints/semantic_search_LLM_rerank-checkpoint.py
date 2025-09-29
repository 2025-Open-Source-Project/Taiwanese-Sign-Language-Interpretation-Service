import torchvision
torchvision.disable_beta_transforms_warning()
from sentence_transformers import util, SentenceTransformer
import pickle
import gc
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

# 清理舊模型與記憶體
try:
    del model
except:
    pass
gc.collect()
torch.cuda.empty_cache()

# 初始化 gte-Qwen2-7B-instruct 模型(只用來做向量檢索)
device = "cuda" if torch.cuda.is_available() else "cpu"
model = SentenceTransformer("Alibaba-NLP/gte-Qwen2-1.5B-instruct", trust_remote_code=True, device=device)

# 載入資料向量
with open("sign_vectors.pkl", "rb") as f:
    data, embeddings, animation_datapath = pickle.load(f)

# 使用者輸入
query = "趕快過馬路"

# 向量化輸入
query_embedding = model.encode(query, convert_to_tensor=True)

# 計算餘弦相似度
cos_scores = util.cos_sim(query_embedding, embeddings)[0]

# 取前五名
top_results = torch.topk(cos_scores, k=5)

print("檢索結果：")
top_sentences = []
for score, idx in zip(top_results.values, top_results.indices):
    sentence = data[idx]
    animation = animation_datapath[idx]
    top_sentences.append(sentence)
    print(f"- 相似度: {score.item():.4f}｜句子: {sentence}｜動畫檔: {animation}")

# --- 下面用 Llama-Breeze2-8B-Instruct 來判斷相似句子 ---

# 初始化 Llama-Breeze2-8B-Instruct (調整成你本地模型路徑或 HF 名稱)
llama_model_name = "MediaTek-Research/Llama-Breeze2-8B-Instruct"
tokenizer = AutoTokenizer.from_pretrained(llama_model_name, trust_remote_code=True)
llama_model = AutoModelForCausalLM.from_pretrained(llama_model_name, device_map="auto", torch_dtype=torch.float16, trust_remote_code=True )

# 產生 prompt，請 LLM 判斷是否有語意相同句子
prompt = f"""
請幫我判斷以下句子是否和輸入句子意思相同：

輸入句子：
「{query}」

候選句子：
"""
for i, sent in enumerate(top_sentences, 1):
    prompt += f"{i}. 「{sent}」\n"

prompt += """
請回答與輸入句子意思相同的句子編號，如果沒有相似的句子，請回答「沒有相似的句子」。
只要給我回答，不要多餘解釋。
"""

# Tokenize 與生成回答
inputs = tokenizer(prompt, return_tensors="pt").to(device)
outputs = llama_model.generate(**inputs, max_new_tokens=50)
answer = tokenizer.decode(outputs[0], skip_special_tokens=True).strip()

print("\nLLM 判斷結果：")
print(answer)
