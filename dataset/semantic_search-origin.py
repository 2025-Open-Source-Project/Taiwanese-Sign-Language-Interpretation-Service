import pickle
import torch
import torchvision
torchvision.disable_beta_transforms_warning()
from sentence_transformers import SentenceTransformer, util
from openai import OpenAI
from openai import APIError, RateLimitError, OpenAIError

# 先初始化本地向量模型
device = "cuda" if torch.cuda.is_available() else "cpu"
model_qwen = SentenceTransformer("Alibaba-NLP/gte-Qwen2-1.5B-instruct", device=device)

# 載入向量資料（之前儲存的 pickle 檔）
with open("sign_vectors.pkl", "rb") as f:
    data, embeddings, animation_paths = pickle.load(f)

# 初始化 OpenAI client 
client = OpenAI(api_key="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")

def query_openai_gpt(messages):
    try:
        response = client.chat.completions.create(
            model="gpt-4.1",
            messages=messages,
            temperature=0.3
        )
        return response.choices[0].message.content
    except RateLimitError as e:
        print("請求頻率限制：", str(e))
    except APIError as e:
        print("API 錯誤：", str(e))
    except OpenAIError as e:
        print("OpenAI SDK 錯誤：", str(e))
    except Exception as e:
        print("未知錯誤：", str(e))
    return None

def find_similar_sentence(query_sentence):
    # 本地化 query embedding
    query_embedding = model_qwen.encode(query_sentence, convert_to_tensor=True)

    # 計算相似度並找 top 5
    cos_scores = util.cos_sim(query_embedding, embeddings)[0]
    top5 = torch.topk(cos_scores, k=5)
    
    print("檢索結果：")
    top_sentences = []
    for score, idx in zip(top5.values, top5.indices):
        sentence = data[idx]
        animation = animation_paths[idx]
        top_sentences.append(sentence)
        print(f"- 相似度: {score.item():.4f}｜句子: {sentence}｜動畫檔: {animation}")
    
    # top_scores = [score.item() for score in top5.values]
    top_animations = [animation_paths[idx] for idx in top5.indices]

    # 準備給 GPT 判斷的 prompt
    system_msg = {
        "role": "system",
        "content": "你是一個語意判斷專家，請判斷下面的句子是否和查詢句意思相同。"
    }
    user_msg = {
        "role": "user",
        "content": (
            f"查詢句: 「{query_sentence}」\n"
            "以下是從資料庫找出的五個最相似句子，請依序回答「意思相同」或「不相同」，"
            "如果有意思相同的，請只列出那些句子本身和對應的動畫檔案路徑；\n"
            "如果沒有意思相同的句子，請回覆：「沒有相似的句子」。\n\n" +
            "\n".join([f"句子：「{sent}」\n動畫路徑：{path}" for i, (sent,  path) in enumerate(zip(top_sentences, top_animations))])
        )
    }

    messages = [system_msg, user_msg]

    # 呼叫 GPT 判斷
    result = query_openai_gpt(messages)
    if result is None:
        return "API 呼叫失敗，無法判斷。"

    return result

if __name__ == "__main__":
    query = "歡迎再度來到台灣"
    answer = find_similar_sentence(query)
    print("GPT 判斷結果：")
    print(answer)

    

