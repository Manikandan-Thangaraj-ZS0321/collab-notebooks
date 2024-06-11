import torch
from PIL import Image
from transformers import AutoModelForCausalLM, AutoTokenizer
from fastapi import FastAPI, UploadFile, File, Form
from fastapi.responses import JSONResponse
import os

app = FastAPI()

MODEL_PATH = "THUDM/cogvlm2-llama3-chat-19B-int4"
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
TORCH_TYPE = torch.bfloat16 if torch.cuda.is_available() and torch.cuda.get_device_capability()[
    0] >= 8 else torch.float16

tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH, trust_remote_code=True)
model = AutoModelForCausalLM.from_pretrained(MODEL_PATH, torch_dtype=TORCH_TYPE, trust_remote_code=True,
                                             low_cpu_mem_usage=True).eval()

text_only_template = "A chat between a curious user and an artificial intelligence assistant. The assistant gives helpful, detailed, and polite answers to the user's questions. USER: {} ASSISTANT:"

history = []


@app.post("/generate")
async def generate(image: UploadFile = File(None), prompt: str = Form(...)):
    global history

    if image:
        image_path = f"/tmp/{image.filename}"
        with open(image_path, "wb") as buffer:
            buffer.write(await image.read())
        image = Image.open(image_path).convert('RGB')
        os.remove(image_path)
    else:
        image = None

    if image is None:
        query = text_only_template.format(prompt) if not history else prompt
        if history:
            old_prompt = ''
            for old_query, response in history:
                old_prompt += old_query + " " + response + "\n"
            query = old_prompt + "USER: {} ASSISTANT:".format(prompt)
        input_by_model = model.build_conversation_input_ids(tokenizer, query=query, history=history,
                                                            template_version='chat')
    else:
        input_by_model = model.build_conversation_input_ids(tokenizer, query=prompt, history=history, images=[image],
                                                            template_version='chat')

    inputs = {
        'input_ids': input_by_model['input_ids'].unsqueeze(0).to(DEVICE),
        'token_type_ids': input_by_model['token_type_ids'].unsqueeze(0).to(DEVICE),
        'attention_mask': input_by_model['attention_mask'].unsqueeze(0).to(DEVICE),
        'images': [[input_by_model['images'][0].to(DEVICE).to(TORCH_TYPE)]] if image is not None else None,
    }

    gen_kwargs = {
        "max_new_tokens": 8192,
        "pad_token_id": 128002,
    }

    with torch.no_grad():
        outputs = model.generate(**inputs, **gen_kwargs)
        outputs = outputs[:, inputs['input_ids'].shape[1]:]
        response = tokenizer.decode(outputs[0])
        print(response)
        response = response.split("")[0]

    history.append((prompt, response))

    return JSONResponse(content={"response": response})


