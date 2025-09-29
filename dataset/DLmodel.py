from transformers import AutoTokenizer, AutoModelForCausalLM

model_name = "TheBloke/Mistral-7B-Instruct-v0.2-GPTQ"
print("正在下載模型，請稍候...")

tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForCausalLM.from_pretrained(model_name, device_map="auto", torch_dtype="auto")
print("模型下載完成！")