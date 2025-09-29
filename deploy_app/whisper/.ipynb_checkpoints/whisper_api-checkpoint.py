import os
os.environ['LD_LIBRARY_PATH'] = '/home/opensource/anaconda3/envs/whisper/lib'
import torch
print(torch.cuda.is_available())

import tensorflow as tf
gpus = tf.config.list_physical_devices('GPU')

if gpus:
    try:
        tf.config.set_logical_device_configuration(
            gpus[0],
            [tf.config.LogicalDeviceConfiguration(memory_limit=4096)]
        )
        logical_gpus = tf.config.list_logical_devices('GPU')
        print(len(gpus), "Physical GPUs,", len(logical_gpus), "Logical GPUs")
    except RuntimeError as e:
        print(e)

from fastapi import FastAPI, File, UploadFile
from fastapi.responses import JSONResponse, RedirectResponse
from fastapi.middleware.cors import CORSMiddleware
from typing import List
import whisper
from snownlp import SnowNLP
import tempfile
from pydub import AudioSegment
import subprocess

torch.cuda.is_available()
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

model = whisper.load_model("medium", device=DEVICE)

# 分段函式：將音訊每一小時切一段（3600 秒）
def split_audio_by_hour(audio_path, output_dir, segment_length=3600):
    audio = AudioSegment.from_file(audio_path)
    duration_seconds = len(audio) // 1000
    segment_paths = []

    for i in range(0, duration_seconds, segment_length):
        start_ms = i * 1000
        end_ms = min((i + segment_length) * 1000, len(audio))
        segment = audio[start_ms:end_ms]

        segment_filename = os.path.join(output_dir, f"segment_{i//segment_length + 1}.mp3")
        segment.export(segment_filename, format="mp3", bitrate="192k")
        segment_paths.append(segment_filename)

    return segment_paths

# 處理整體流程
def transcribe_large_audio(audio_path):
    with tempfile.TemporaryDirectory() as temp_dir:
        # 1. 音檔分段（每段 1 小時）
        segments = split_audio_by_hour(audio_path, temp_dir)

        # 2. 每段送入 Whisper 模型
        all_text = ""
        for i, segment_path in enumerate(segments):
            print(f"Transcribing segment {i+1}/{len(segments)}: {segment_path}")
            result = model.transcribe(
                segment_path, 
                language="zh",
                task="transcribe",
                initial_prompt="請將以下語音轉成帶有完整標點符號的中文句子。"
            )
            all_text += f"{result['text']}"

        return all_text


@app.post("/transcribe/")
async def transcribe_multiple_files(files: List[UploadFile] = File(...)):
    result = []

    for audio_file in files:
        print(f"Processing file: {audio_file.filename}")
        # 儲存為臨時檔案
        with tempfile.NamedTemporaryFile(delete=False, suffix=os.path.splitext(audio_file.filename)[-1]) as temp:
            temp.write(await audio_file.read())
            temp_path = temp.name

        # 執行語音辨識
        try:
            transcription = transcribe_large_audio(temp_path)
            print("Whisper 辨識結果：", transcription)
            result.append({
                "file_name": audio_file.filename,
                "transcription": transcription
            })
        except Exception as e:
            result.append({
                "file_name": audio_file.filename,
                "error": str(e)
            })
        finally:
            os.remove(temp_path)  # 清除暫存檔案

    return JSONResponse(content={"result": result})

@app.get("/", response_class=RedirectResponse)
async def redirect_to_docs():
    return "/docs"
