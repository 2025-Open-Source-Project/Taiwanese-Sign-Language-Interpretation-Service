import requests
import json
from bs4 import BeautifulSoup
from tqdm import tqdm
import time

# 設定起始與結束的 serno 值
start_serno = 1
end_serno = 100

# 結果儲存
results = []
count = 1

# 主迴圈
for serno in tqdm(range(start_serno, end_serno + 1)):
    url = f"https://jung-hsingchang.tw/name/areavideo.php?serno={serno}"
    
    try:
        response = requests.get(url, timeout=5)
        response.encoding = 'utf-8'  # 強制使用 UTF-8 編碼
        if response.status_code != 200 or 'video' not in response.text:
            continue  # 網頁不存在或沒有內容就跳過

        soup = BeautifulSoup(response.text, "html.parser")

        # 嘗試擷取資料
        name_tag = soup.find('td', colspan="5")
        description_tag = soup.find_all("span", class_="style7")[-1]
        video_tag = soup.find("video")
        source_tag = video_tag.find("source") if video_tag else None

        result = {
            "count": f"{count:04}",
            "id": serno,
            "text": name_tag.text.strip() if name_tag else None,
            "sign_animation":  source_tag['src'][2:] if source_tag else None
        }
        count += 1
        results.append(result)

    except Exception as e:
        # 如果有錯誤就跳過
        continue

# 儲存為 JSON 檔案
with open("tsl_place_signs.json", "w", encoding="utf-8") as f:
    json.dump(results, f, ensure_ascii=False, indent=2)

print("已完成，資料數量：", len(results))
