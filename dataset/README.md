### How to run
**Use tmux to work continuously**
`uvicorn semantic_search:app --host 0.0.0.0 --port 9000`

### 主要程式碼
1. semantic_search.py - 負責語意相似度判斷
+ 分句處理
+ 以 gte-qwen2 進行 RAG 檢索: 快速從進 5000 筆的資料中取得與查詢句最相似的 5 筆資料  

![](https://raw.githubusercontent.com/2025-Open-Source-Project/Taiwanese-Sign-Language-Interpretation-Service/main/dataset/%E8%9E%A2%E5%B9%95%E6%93%B7%E5%8F%96%E7%95%AB%E9%9D%A2%202025-09-17%20004814.png)

+ 以 mistral-7B 做精確語意判斷: 從選出的 5 筆資料判斷有無與查詢句完全意思相同者  

![](https://raw.githubusercontent.com/2025-Open-Source-Project/Taiwanese-Sign-Language-Interpretation-Service/main/dataset/%E8%9E%A2%E5%B9%95%E6%93%B7%E5%8F%96%E7%95%AB%E9%9D%A2%202025-09-17%20012153.png)  

+ 將結果(包含相似句子與動畫路徑)回傳給前端   

~~About 30 sec every search (Gemma 4 min)~~  
**About 3 sec every search (Mistral-7B)**

### Data Flow  
文字資料輸入 -> 語意檢索（使用 FAISS 與中文大語言模型 Qwen2-1.5B）-> 語意判斷（開源 chat 模型 Mistral）-> 結果傳至前端

### Open Source models
+ embedding search model link (gte-qwen2-1.5B-it): https://huggingface.co/Alibaba-NLP/gte-Qwen2-1.5B-instruct
+ chat LLM mistral model link: https://huggingface.co/TheBloke/Mistral-7B-Instruct-v0.2-GPTQ

### Deployed in http://140.123.105.233:9000 (need CCU VPN to connect)

### Others
+ rm hugging face model in cache -> download model again from hugging face repo every reload
 1. go `~/.cache/huggingface/hub`
 
+ now completely deploy in local !
