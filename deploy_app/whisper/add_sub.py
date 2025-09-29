import whisper
import srt
import datetime

# 1. 載入模型
model = whisper.load_model("base")

# 2. 轉錄音訊
result = model.transcribe("Demo-video.mp4")

# 3. 產生 SRT 字幕段
subs = []
for i, segment in enumerate(result['segments']):
    start = datetime.timedelta(seconds=segment['start'])
    end = datetime.timedelta(seconds=segment['end'])
    content = segment['text'].strip()
    subs.append(srt.Subtitle(index=i+1, start=start, end=end, content=content))

# 4. 寫入 .srt 檔案
with open("demo_video_sub.srt", "w", encoding="utf-8") as f:
    f.write(srt.compose(subs))
