# Taiwanese-Sign-Language-Interpretation-Service

## 簡介
   為了推動資訊平權並協助聽障者獲取日常語音資訊，我們開發了一套即時生成臺灣手語動畫的服務，讓使用者能夠即時將語音或文字內容轉換為符合臺灣手語語法的動畫，無需仰賴真人手語翻譯。
   系統流程結合了語音轉文字（OpenAI Whisper）、語意檢索（使用 FAISS 與中文大語言模型 Qwen2-1.5B）、語意判斷與生成（開源 chat 模型 Mistral-7B）以及動畫生成（Blender 動畫渲染），打造出從語音輸入到手語動畫輸出的完整 pipeline model。  
   除此之外，我們部署服務到 Kserve 雲端原生平台部署系統，以實現模組化、可擴充的即時服務，具備彈性調度、容易維運的特性。未來，在持續擴增資料集後可廣泛應用於日常生活，補足現有資訊傳遞的斷層，讓手語翻譯服務能真正走入生活。  

![](https://raw.githubusercontent.com/2025-Open-Source-Project/Taiwanese-Sign-Language-Interpretation-Service/main/whole_process.gif)


## 理論基礎
    我們運用了自然語言處理與語音辨識技術，開發即時手語翻譯(生成)服務系統。使用者透過網頁前端輸入文字、上傳音訊檔案，或進行即時錄音，系統首先利用 Whisper 模型進行語音辨識，將音訊內容準確轉換為文字資料。
    接著，這些文字會被傳到後端的語意檢索模組，先由 gte-Qwen-1.5B 向量模型將輸入句子編碼為語意向量，並與資料庫中預先向量化的手語資料進行語意相似度計算，篩選出語意最接近的前五筆資料。之後，我們再透過本地部署的 chat 語言模型，對這些相似候選句進行進一步語意精確比對與語義一致性驗證，以判斷是否存在與輸入語意完全相同的語句。
    最後，若最終比對結果符合，我們即從資料庫中擷取對應的手語動畫檔案，傳回至前端展示；若無語意一致之句子，則回傳「沒有相似的句子」，確保翻譯品質與用戶體驗。  

![](https://raw.githubusercontent.com/2025-Open-Source-Project/Taiwanese-Sign-Language-Interpretation-Service/main/structure.png)

## 系統流程
![](https://raw.github.com/2025-Open-Source-Project/Taiwanese-Sign-Language-Interpretation-Service/main/data_flow.png)

## 設計創新說明
    台灣目前網路上關於本土手語的電子資源並不多，大部分的查詢資源距今都已有十多年，之前的資源多有些過時，彼此間的語法也不統一。    
    在著手製作這套系統前，我們在搜尋相關資料中，不論國內外找到了許多手語翻譯系統，但卻都沒有成功普及於生活中。即便在開始注重資訊平權的現在，缺乏手語人才時，手語翻譯系統仍然未受到重用。為探究其原因，我們諮詢了台灣手語的專家，並得到了以下現在手語翻譯系統碰到的問題:  
        (1) 台灣手語還是一個「發展中的語言」，目前為止就連專家都還需要透過與聽障朋友交流發掘新的手語表示。  
        (2) 手語是一個具有空間性的語言，與平時所說的說話不同，相同的語句並不能通過替換主詞、名詞等擴增資料或以此類推。  

    舉例來說，「我和你吃飯」跟「我和她吃飯」是需要對話人物在空間中的位置才可以表現的(前者需指向面向的人;後者需指向指名的人)。再舉動詞的使用為例，光是「吃」一個字在手語中就有超過一千多種比法，並且還不是全部都有紀錄，根據食物的不同而異。  

    於是，為了應對上述問題，我們決定先以可以產生準確連貫的手語表示為目標。我們因此採取以下方法:  
        (1) 以語意比對先做到正確翻譯與動畫展現，而非直接用機器學習翻譯。機器學習的方法雖然成功的話泛用性會較廣，但以目前的狀況來說，產出的多是無法採用的手語輸出。  
        (2) 為了彌補資料不足的困境，我們將資料庫設計的容易加入新資料，不需因新資料的加入重新更新整個資料庫，並鼓勵有條件的使用者貢獻所擁有的手語資料，使開放服務更具使用性。  


## 使用環境
+ 作業系統: Linux
+ 開發環境: JupyterNotebookServer, 動畫: 本地端 VS code
+ 程式語言: python
+ 使用軟體: Whisper, Openpose, Blender, Mistral, gte-qwen2, FAISS, Kserve, MocapNET, 3DPoseTracker
+ 硬體: 
    + CPU：12th Gen Intel(R) Core(TM) i7-12650H   2.30 GHz
    + GPU：NVIDIAGeForceRTX3090 * 2

## 前置安裝
```
- accelerate == 1.0.1
- fastapi >= 0.115.13
- huggingface-hub >= 0.32.0
- sentence_transformers >=  3.2.1  
- torch == 2.0.1+cu117 
- torchvision == 0.15.2
- mtkresearch >= 0.3.1
- transformer >= 4.46.3 
- timm >= 1.0.15 
- tqdm >= 4.67.1                   
- numpy >= 1.24.4 
- einops >=  0.8.1  
- openai >= 1.88.0
- pydantic >= 2.10.6 
- fastapi >= 0.115.13 
- uvicorn == 0.33.0 
- aiofiles >= 24.1.0
- python-multipart >=  0.0.20  
- openai-whisper == 20240930 
- pydub >= 0.25.1  
- ffmpeg >= 1.4 
```
## 模型測試
+ 連接 CCU VPN
+ 到 `dataset/test/`
+ 安裝 `selenium` -> `pip install selenium`
+ 可以自主更新 test_datafile，詳細運作方式請見 [dataset/test/README](https://github.com/2025-Open-Source-Project/Taiwanese-Sign-Language-Interpretation-Service/tree/d6d4e0490d8792c15c44271e4cfba45286c2e691/dataset/test)
+ 執行: `python test.py`
  + 結束後會產生混淆矩陣與錯誤點的txt檔

## 功能說明
+ [website](https://github.com/2025-Open-Source-Project/Taiwanese-Sign-Language-Interpretation-Service/tree/d6d4e0490d8792c15c44271e4cfba45286c2e691/website): 管理使用者唯一直接接觸到系統的部分，會負責讓使用者輸入欲查詢的資料，並輸出手語影片或查詢失敗。
+ [whisper](https://github.com/2025-Open-Source-Project/Taiwanese-Sign-Language-Interpretation-Service/tree/d6d4e0490d8792c15c44271e4cfba45286c2e691/deploy_app/whisper): 利用 Open AI 的開源軟體 Whisper 將音檔或使用者的錄音轉換為文字，並傳遞資料給語意判斷模組(Qwen2 + mistral)。
+ [dataset](https://github.com/2025-Open-Source-Project/Taiwanese-Sign-Language-Interpretation-Service/tree/d6d4e0490d8792c15c44271e4cfba45286c2e691/dataset): 將傳過來的內容切割處理，先經過 qwen2-1.5B 做語意向量檢索，從我們的向量資料庫中找出前五名與查詢語句最相近的資料，再將五筆資料交給 Mistral 做精確判斷(相同與否)。
+ [videos](https://github.com/2025-Open-Source-Project/Taiwanese-Sign-Language-Interpretation-Service/tree/d6d4e0490d8792c15c44271e4cfba45286c2e691/website/static/videos/): 儲存 Blender 手語動畫，提供影片給前端顯示

## 文件結構
```bash
Taiwanese-Sign-Language-Interpretation-Service
├── README.md
├── anime
│   ├── README.md
│   └── output_jsons
├── data_flow.png
├── dataset
│   ├── DLmodel.py
│   ├── README.md
│   ├── crew_data
│   │   ├── README.md
│   │   ├── crewer_for_jsn.py
│   │   ├── crewer_for_jsn_sen.py
│   │   ├── crewer_place_jsn.py
│   │   ├── tsl_signs.json
│   │   └── tsl_signs_sen.json
│   ├── filtered_file.json
│   ├── full_dataset.json
│   ├── gen_video.py
│   ├── gte-Qwen2-1.5B
│   ├── gte-Qwen2-7B
│   ├── gte-qwen_search_result.png
│   ├── mistral_determine_result.png
│   ├── models
│   │   ├── mistral
│   │   │   ├── README.md
│   │   │   ├── config.json
│   │   │   ├── generation_config.json
│   │   │   ├── quantize_config.json
│   │   │   ├── special_tokens_map.json
│   │   │   ├── tokenizer.json
│   │   │   ├── tokenizer.model
│   │   │   └── tokenizer_config.json
│   │   └── qwen
│   │       ├── 1_Pooling
│   │       │   └── config.json
│   │       ├── README.md
│   │       ├── added_tokens.json
│   │       ├── config.json
│   │       ├── config_sentence_transformers.json
│   │       ├── generation_config.json
│   │       ├── merges.txt
│   │       ├── model.safetensors.index.json
│   │       ├── modeling_qwen.py
│   │       ├── modules.json
│   │       ├── scripts
│   │       │   └── eval_mteb.py
│   │       ├── sentence_bert_config.json
│   │       ├── special_tokens_map.json
│   │       ├── tokenization_qwen.py
│   │       ├── tokenizer.json
│   │       ├── tokenizer_config.json
│   │       └── vocab.json
│   ├── rag_test
│   │   ├── comp.py
│   │   ├── faiss_emb.py
│   │   ├── faiss_query.py
│   │   ├── sign_index_cosine.faiss
│   │   └── untitled.py
│   ├── requirements.txt
│   ├── semantic_search-origin.py
│   ├── semantic_search.py
│   ├── semantic_search_LLM_rerank.py
│   ├── semantic_search_only.py
│   ├── sentence_embedding.py
│   ├── test
│   │   ├── 1st
│   │   │   ├── frontend_confusion_matrix_20250916_172049.png
│   │   │   └── frontend_error_cases_20250916_172428.txt
│   │   ├── 2nd
│   │   │   ├── frontend_confusion_matrix_20250923_212534.png
│   │   │   └── frontend_error_cases_20250923_212606.txt
│   │   ├── README.md
│   │   ├── err_msg_example.png
│   │   ├── test.py
│   │   └── test_datafile.json
│   └── trans_cache_to_local_dir.py
├── deploy_app
│   ├── README.md
│   └── whisper
│       ├── README.md
│       ├── add_sub.py
│       ├── requirements.txt
│       └── whisper_api.py
├── sign_animation
│   ├── MakeHuman_avatar.png
│   ├── Mixamo_avatar(w).blend
│   ├── README.md
│   ├── body_scale_result_in_dif_poses.png
│   ├── livelastRun3DHiRes.mp4
│   ├── miximo_avatar.blend
│   ├── motion_new.bvh
│   ├── out.bvh
│   ├── rokoko-studio-live-blender-master.zip
│   ├── think_2.bvh
│   ├── ThreeDPoseTracker_TDPT_Win_x64_v0_6_2
│   └── typhoon_day.bvh
├── structure.png
├── website
│   ├── README.md
│   ├──  static
│   │   ├── index.html
│   │   ├── videos
│   │   │   ├── sentence
│   │   │   │   └── t
│   │   │   │       └── S-think_2.mp4
│   │   │   └── video
│   │   └── videos.json
│   ├── webpage_api.py
│   └── webpage_fig.png
├── deprecated_model
└── whole_process.gif
```
