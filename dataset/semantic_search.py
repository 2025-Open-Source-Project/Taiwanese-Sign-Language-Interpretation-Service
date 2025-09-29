import pickle
import torch
import torchvision
torchvision.disable_beta_transforms_warning()
from sentence_transformers import SentenceTransformer, util
from pydantic import BaseModel
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
# from pathlib import Path

# 先初始化本地向量模型
device = "cuda" if torch.cuda.is_available() else "cpu"

# qwen_path = Path.home() / "text2anime-proj/dataset/models/Alibaba-NLP/gte-Qwen2-1.5B-instruct"
qwen_path = "models/qwen"
model_qwen = SentenceTransformer(qwen_path, device=device, local_files_only=True)

# 載入向量資料（之前儲存的 pickle 檔）
with open("sign_vectors.pkl", "rb") as f:
    sentences, embeddings, animation_paths = pickle.load(f)

app = FastAPI()   
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

from transformers import AutoTokenizer, AutoModelForCausalLM

# model_mistral = os.path.join(home_dir, "text2anime-proj/dataset/models/TheBloke/Mistral-7B-Instruct-v0.2-GPTQ")
model_mistral = "models/mistral"
tokenizer = AutoTokenizer.from_pretrained(model_mistral, use_fast=True)
model = AutoModelForCausalLM.from_pretrained(
    model_mistral,
    device_map="auto",
    torch_dtype=torch.float16,
    low_cpu_mem_usage=True,
)

import re
# 分句
def split_into_sentences(text):
    sentences = re.split(r'(?<=[。！？\n\r\t])', text) #，,
    sentences = [s.strip(" ，,\n\r\t") for s in sentences if s.strip(" \n\r\t")] #，,
    print("分句結果：", sentences)
    return sentences

def query_mistral(usr_prompt, query_sentence):
   
    inputs = tokenizer(usr_prompt, return_tensors="pt").to(model.device)
    outputs = model.generate(
        **inputs,
        max_new_tokens=256,
        do_sample=False,
        pad_token_id=tokenizer.eos_token_id,
        # eos_token_id=tokenizer.convert_tokens_to_ids("<end_of_turn>")
    )
    whole_ans = tokenizer.decode(outputs[0], skip_special_tokens=True)
    
    print("Mistral 原始輸出：", whole_ans)
    
    result = re.search(r"<end>\s*(.+?)(相似句子[:：].+?)<end>", whole_ans, re.DOTALL)
    
    # print(result.group(2))

    if result:
        match = re.search(
            r"相似句子[:：]\s*(.+?)\s*[\n;；]\s*路徑[:：]?\s*(\S+)",
            result.group(2),
            re.DOTALL
        )
        if match:
            sentence = match.group(1).strip()
            path = match.group(2).strip()
            return {
                "sentence": query_sentence,
                "animation_path": path
            }
        
    if "沒有相似的句子" in whole_ans:
        return {
            "sentence": query_sentence,
            "animation_path": "沒有相似的句子"
        }


    return "格式無法解析，請確認模型輸出"


def find_similar_sentence(query_sentence):
    # 本地化 query embedding
    query_embedding = model_qwen.encode(query_sentence, convert_to_tensor=True)
    
    # 計算相似度並找 top 5
    cos_scores = util.cos_sim(query_embedding, embeddings)[0]
    # print("in embeddings")
    top5 = torch.topk(cos_scores, k=5)
    
    print("檢索結果：")
    top_sentences = []
    for score, idx in zip(top5.values, top5.indices):
        sentence = sentences[idx]
        animation = animation_paths[idx]
        top_sentences.append(sentence)
        print(f"- 相似度: {score.item():.4f}｜句子: {sentence}｜動畫檔: {animation}")
    
    # top_scores = [score.item() for score in top5.values]
    top_animations = [animation_paths[idx] for idx in top5.indices]

    # text prompt
    usr_prompt = (
        f"你是一個語意判斷專家，負責判斷資料庫中的句子是否與查詢句意思完全相同。\n"
        f"查詢句: 「{query_sentence}」\n\n"
        "以下是從資料庫找出的五個最相似句子，請依序查看是否有與查詢句意思完全相同的句子。\n"
        "回覆規則：\n"
        "1. 如果資料庫中的五個最相似句子有與查詢句意思完全相同的句子，請從提供的資料庫句子中**只選出一個**完全相同的，並如此回覆：\n"
        '相似句子: <句子>； 路徑: <動畫路徑> <end>\n'
        "2. 如果沒有意思完全相同的句子，請回覆：\n"
        '沒有相似的句子\n'
        "3. 不得輸出任何其他文字或解釋。\n"
        "4. 回答的句子與路徑完全為資料庫內容，不可任意組合。\n\n"
        "資料庫句子：\n" +
        "\n".join([f'句子: 「{sent}」\n動畫路徑: {path}' for sent, path in zip(top_sentences, top_animations)]) +
        "\n<end>"
    )

    result = query_mistral(usr_prompt, query_sentence)
    
    print("Mistral 回覆：\n", result)
    
    if result is None:
        return "API 呼叫失敗，無法判斷。"

    return result

class TextInput(BaseModel):
    text: str

@app.post("/semantic_check")
async def semantic_check(input: TextInput):
    try:
        results = []
        sentences = split_into_sentences(input.text)
        for sentence in sentences:
            print(f"now processing sentence...{sentence}")
            result = find_similar_sentence(sentence)
            # print(sentence, "=>", result)
            if result:
                results.append(result)
            else:
                results.append({
                    "sentence": sentence,
                    "animation_path": "沒有相似的句子"
                })
        return {"result": results}
    except Exception as e:
        return {"error": str(e)}
    

