### How to run
`uvicorn semantic_search:app --host 0.0.0.0 --port 9000`

~~**About 30 sec every search (Gemma 4 min)**~~
**About 3 sec every search (Mistral-7B)**

+ embedding search model link (gte-qwen2-1.5B-it): https://huggingface.co/Alibaba-NLP/gte-Qwen2-1.5B-instruct
+ mistral model link: https://huggingface.co/TheBloke/Mistral-7B-Instruct-v0.2-GPTQ

in http://140.123.105.233:9000

+ rm hugging face model in cache -> download model again from hugging face repo every reload
 1. go `~/.cache/huggingface/hub`
 
+ now completely in local !
