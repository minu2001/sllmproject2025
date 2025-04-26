from fastapi import FastAPI
from pydantic import BaseModel
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
import os
from fastapi.middleware.cors import CORSMiddleware



app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


os.environ['HF_TOKEN'] = "-----"
token = os.environ["HF_TOKEN"]

bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_compute_dtype=torch.float16,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_use_double_quant=True,
    llm_int8_enable_fp32_cpu_offload=True
)

device_map = {
    "model.embed_tokens": "cuda:0",
    "model.norm": "cuda:0",
    "lm_head": "cuda:0",

}


tokenizer = AutoTokenizer.from_pretrained('openchat/openchat-3.5-1210', token=token)
model = AutoModelForCausalLM.from_pretrained(
    'openchat/openchat-3.5-1210',
    token=token,
    quantization_config=bnb_config,
    device_map="auto",
    torch_dtype=torch.float16,
    low_cpu_mem_usage=True,
    offload_folder="offload_folder"
)




class Message(BaseModel):
    role: str
    content: str


class ChatRequest(BaseModel):
    model: str
    messages: list[Message]


@app.post("/v1/chat/completions")
async def chat_completion(request: ChatRequest):
    input_text = request.messages[-1].content
    input_ids = tokenizer(input_text, return_tensors="pt").to("cuda")

    # 생성 시 메모리 제한 설정
    with torch.no_grad():
        output_ids = model.generate(
            **input_ids,
            do_sample=True,
            temperature=0.7,
            max_new_tokens=1024
        )

    output_text = tokenizer.decode(output_ids[0], skip_special_tokens=True)
    return {"choices": [{"message": {"role": "assistant", "content": output_text}}]}


@app.get("/")
def root():
    return {"message": "Server is running!"}


if __name__ == "__main__":

    torch.cuda.empty_cache()
    import uvicorn

    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)