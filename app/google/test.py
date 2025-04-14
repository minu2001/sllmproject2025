import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
import os

os.environ['HF_TOKEN'] = "----"

tokenizer = AutoTokenizer.from_pretrained('google/gemma-2-2b-it')
model = AutoModelForCausalLM.from_pretrained('google/gemma-2-2b-it')

input_text = "이순신에 대해서 알려줘잉?"
input_ids = tokenizer(input_text, return_tensors="pt")

outputs = model.generate(**input_ids, max_length=512)
print(tokenizer.decode(outputs[0]))