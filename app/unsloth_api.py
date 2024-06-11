import torch
from fastapi import FastAPI, Form
import os
import gc
from unsloth import FastLanguageModel


app = FastAPI()
max_seq_length = 2048
dtype = None
load_in_4bit = True

unsloth_model = "/home/hera/workspace/llama3/unsloth/models--unsloth--llama-3-8b-Instruct-bnb-4bit/snapshots/efa44c86af4fcbbc3d75e6cb1c8bfaf7f5c7cfc1"

model, tokenizer = FastLanguageModel.from_pretrained(
    model_name=unsloth_model,
    max_seq_length=max_seq_length,
    dtype=dtype,
    load_in_4bit=load_in_4bit,
)


@app.post("/generate")
async def generate(ocr: str = Form(...), prompt: str = Form(...)):

    try:
        messages = [
            {"role": "system", "content": prompt},
            {"role": "user", "content": ocr},
            {"role": "assistant:", "content": ""},
        ]

        prompt = tokenizer.apply_chat_template(
            messages,
            tokenize=True,
            add_generation_prompt=True,
            return_tensors="pt",
            map_eos_token=True,
        ).to("cuda")

        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token

        outputs = model.generate(input_ids=prompt, max_new_tokens=8192, use_cache=True,
                                 pad_token_id=tokenizer.pad_token_id)

        generated_responses = tokenizer.batch_decode(outputs, skip_special_tokens=True)

        assistant_response = generated_responses[0]

        return assistant_response

    except Exception as ex:
        raise ex
    finally:
        gc.collect()
        torch.cuda.empty_cache()

