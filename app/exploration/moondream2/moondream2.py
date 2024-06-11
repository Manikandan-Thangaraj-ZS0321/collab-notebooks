import gc

from transformers import AutoModelForCausalLM, AutoTokenizer
from PIL import Image
from fastapi import FastAPI, UploadFile, File, Form

import torch
import os


app = FastAPI()

model_id = "vikhyatk/moondream2"
revision = "2024-05-20"

TORCH_TYPE = torch.bfloat16 if torch.cuda.is_available() and torch.cuda.get_device_capability()[0] >= 8 else torch.float16

tokenizer = AutoTokenizer.from_pretrained(model_id, trust_remote_code=True)
model = AutoModelForCausalLM.from_pretrained(model_id, torch_dtype=TORCH_TYPE, trust_remote_code=True,
                                             low_cpu_mem_usage=True).eval()
model.to('cuda')


@app.post("/generate")
async def generate(image: UploadFile = File(None), prompt: str = Form(...)):
    try:
        if image:
            image_path = f"/tmp/{image.filename}"
            with open(image_path, "wb") as buffer:
                buffer.write(await image.read())
            image = Image.open(image_path)
            os.remove(image_path)
        else:
            image = None

        enc_image = model.encode_image(image)
        answer = model.answer_question(enc_image, prompt, tokenizer)
        print(answer)
        return answer
    finally:
        gc.collect()
        torch.cuda.empty_cache()


