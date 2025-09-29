import json
from pathlib import Path

# 檔案路徑
input_path = Path("filtered_file.json")   # 你的資料庫 JSON
output_path = Path("videos.json")        # 要輸出的 JSON

# 載入資料庫
with open(input_path, "r", encoding="utf-8") as f:
    data = json.load(f)

# 擷取 sign_animation 欄位
animations = []
for item in data:
    if "sign_animation" in item:
        animations.append(item["sign_animation"])

# 去掉重複並排序
animations = sorted(set(animations))

# 寫出 videos.json
with open(output_path, "w", encoding="utf-8") as f:
    json.dump({"videos": animations}, f, ensure_ascii=False, indent=2)

print(f"已輸出 {len(animations)} 個動畫路徑到 {output_path}")

