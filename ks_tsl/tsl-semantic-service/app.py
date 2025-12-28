import os
import re
import pickle
import torch

from sentence_transformers import SentenceTransformer, util
from transformers import AutoTokenizer, AutoModelForCausalLM
from pydantic import BaseModel
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

# ========= 設定路徑 / device（改用環境變數） =========

device = "cuda:0" if torch.cuda.is_available() else "cpu"
qwen_device = os.getenv("QWEN_DEVICE", "cpu")

SIGN_VECTORS_PATH = os.getenv("SIGN_VECTORS_PATH", "/mnt/models/sign_vectors.pkl")
QWEN_MODEL_PATH = os.getenv("QWEN_MODEL_PATH", "/mnt/models/models/gte-Qwen2-1.5B-instruct")
MISTRAL_MODEL_PATH = os.getenv("MISTRAL_MODEL_PATH", "/mnt/models/models/Mistral-7B-Instruct-v0.2")

print(f"Using device: {device}, qwen_device: {qwen_device}")
print(f"SIGN_VECTORS_PATH={SIGN_VECTORS_PATH}")
print(f"QWEN_MODEL_PATH={QWEN_MODEL_PATH}")
print(f"MISTRAL_MODEL_PATH={MISTRAL_MODEL_PATH}")

# ========= 載入 Qwen embedding model =========

model_qwen = SentenceTransformer(QWEN_MODEL_PATH, device=qwen_device)

# ========= 載入向量資料 =========

with open(SIGN_VECTORS_PATH, "rb") as f:
    sentences, embeddings, animation_paths = pickle.load(f)

    if isinstance(embeddings, torch.Tensor):
        embeddings = embeddings.to("cpu")
    elif isinstance(embeddings, list) and isinstance(embeddings[0], torch.Tensor):
        embeddings = torch.stack([e.to("cpu") for e in embeddings], dim=0)

print(f"Loaded {len(sentences)} sentences from sign_vectors.pkl")

# ========= FastAPI 初始化 =========

app = FastAPI()
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ========= 載入 Mistral =========

tokenizer = AutoTokenizer.from_pretrained(MISTRAL_MODEL_PATH, use_fast=True)
model = AutoModelForCausalLM.from_pretrained(
    MISTRAL_MODEL_PATH,
    device_map={"": 0},
    torch_dtype=torch.float16,
    low_cpu_mem_usage=True,
)
model.to(device)
model.eval()

# ========= 工具函式 =========

def split_into_sentences(text: str):
    sents = re.split(r'(?<=[ 。！？\n\r\t])', text)
    sents = [s.strip(" ，,\n\r\t") for s in sents if s.strip(" \n\r\t")]
    print("分句結果：", sents)
    return sents

def query_mistral(usr_prompt, query_sentence, top_sentences, top_animations):
    inputs = tokenizer(usr_prompt, return_tensors="pt").to(model.device)
    outputs = model.generate(
        **inputs,
        max_new_tokens=8,
        do_sample=False,
        pad_token_id=tokenizer.eos_token_id,
    )
    whole_ans = tokenizer.decode(outputs[0], skip_special_tokens=True).strip()
    print("Mistral 原始輸出：", whole_ans)

    result = re.search(r"<end>\s*(.*)", whole_ans, re.DOTALL)
    if result:
        match = re.search(r"\s*(\d+)", result.group(1), re.DOTALL)
        if match and match.group(1).isdigit():
            idx = int(match.group(1)) - 1
            if 0 <= idx < 5:
                path = top_animations[idx]
                sentence = top_sentences[idx]
                return {"sentence": sentence, "animation_path": path}

        # 沒有合法數字，就看有沒有「沒有句子」
        if "沒有句子" in result.group(1):
            return {
                "sentence": query_sentence,
                "animation_path": "沒有相似的句子",
            }
        return {
            "sentence": query_sentence,
            "animation_path": "沒有相似的句子",
        }

    # 完全沒 match 的 fallback
    return {
        "sentence": query_sentence,
        "animation_path": "沒有相似的句子",
    }

def find_similar_sentence(query_sentence: str):
    # 1. query embedding
    query_embedding = model_qwen.encode(query_sentence, convert_to_tensor=True)

    # 2. cos 相似度 + top 5
    cos_scores = util.cos_sim(query_embedding, embeddings)[0]
    top5 = torch.topk(cos_scores, k=5)

    print("檢索結果：")
    top_sentences = []
    for score, idx in zip(top5.values, top5.indices):
        sentence = sentences[idx]
        animation = animation_paths[idx]
        top_sentences.append(sentence)
        print(f"- 相似度: {score.item():.4f}｜句子: {sentence}｜動畫檔: {animation}")

    top_animations = [animation_paths[idx] for idx in top5.indices]

    # 3. 組 prompt 給 Mistral 判斷
    usr_prompt = (
        f"你是一個語意判斷專家，負責判斷資料庫中的句子是否與查詢句意思完全相同且涵蓋其意境。\n"
        f"查詢句: 「{query_sentence}」\n\n"
        "以下是從資料庫找出的五個最相似句子，請依序查看是否有與查詢句意思完全相同且能完全涵蓋的句子。\n"
        "回覆規則：\n"
        "1. 如果有符合的句子，請只回覆該句子的編號 (1~5)。\n"
        "2. 如果沒有句意相同的句子，請回覆：\n"
        "沒有句子\n"
        "3. 不得輸出任何其他文字或解釋。\n\n"
        "候選句子：\n"
        + "\n".join([f"({i+1}) {sent}" for i, sent in enumerate(top_sentences)])
        + "\n<end>"
    )

    result = query_mistral(usr_prompt, query_sentence, top_sentences, top_animations)
    print("Mistral 回覆：\n", result)

    if result is None:
        return {
            "sentence": query_sentence,
            "animation_path": "API 呼叫失敗，無法判斷。",
        }
    return result

# ========= Pydantic model & API =========

class TextInput(BaseModel):
    text: str

@app.post("/semantic_check")
async def semantic_check(input: TextInput):
    try:
        results = []
        sents = split_into_sentences(input.text)
        for sentence in sents:
            print(f"now processing sentence...{sentence}")
            result = find_similar_sentence(sentence)
            if result:
                results.append(result)
            else:
                results.append(
                    {"sentence": sentence, "animation_path": "沒有相似的句子"}
                )
        return {"result": results}
    except Exception as e:
        return {"error": str(e)}

@app.get("/health")
async def health():
    return {"status": "ok"}
