import requests
import json

output = []
count = 1

for id_num in range(0, 5001):
    base_url = "https://twtsl.ccu.edu.tw/TSL//lib/api.php"
    
    # API 1: querySearch
    query_url = f"{base_url}?fname=querySearch&id={id_num}&lang=zh_tw"
    try:
        r1 = requests.get(query_url)
        if r1.status_code != 200:
            continue
        try:
            data1 = r1.json()
        except ValueError:
            continue
        if not isinstance(data1, dict) or "Record" not in data1 or not data1["Record"]:
            continue
        record = data1["Record"][0]
        clip_raw = record.get("clip")
        if not clip_raw:
            continue
        clip_name = clip_raw.replace("video/t/", "") + ".mp4"
        sign_id = record.get("id")
    except Exception:
        continue

    # API 2: group
    group_url = f"{base_url}?fname=group&id={id_num}&lang=zh_tw"
    try:
        r2 = requests.get(group_url)
        if r2.status_code != 200:
            continue
        try:
            data2 = r2.json()
        except ValueError:
            continue
        if not isinstance(data2, dict) or "Record" not in data2 or not data2["Record"]:
            continue
        for name_entry in data2["Record"]:
            text = name_entry.get("name")
            if text:
                output.append({
                    "count": f"{count:04}",
                    "id": sign_id,
                    "text": text,
                    "sign_animation": clip_name
                })
                count += 1
    except Exception:
        continue

# 儲存為 JSON 檔案
with open("tsl_signs.json", "w", encoding="utf-8") as f:
    json.dump(output, f, ensure_ascii=False, indent=2)

print("已完成，資料數量：", len(output))
