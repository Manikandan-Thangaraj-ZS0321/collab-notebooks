import torch
import gc

from pydantic import BaseModel
from fastapi import FastAPI
from model_load import ModelLoad

pipeline = ModelLoad.krypton_chat_llamacpp_model_load()
app = FastAPI()
paddle_ocr = ModelLoad.paddleocr_model_load()


class LlamaRequest(BaseModel):
    inputFilePath: str
    promptFilePath: str


def get_file_content(file_path):
    with open(file_path, 'r') as file:
        text = file.read()
        return text


def generate_tokens_paddle(image_path: str) -> str:
    try:
        result_paddle = paddle_ocr.ocr(image_path, cls=True)
        extracted_text = ""
        for result in result_paddle:
            for record in result:
                txt = record[1][0]
                extracted_text += txt + "\n"
        return extracted_text
    except Exception as e:
        raise e


@app.post("/chat/krypton")
def read_item(request: LlamaRequest):
    try:
        ocr_result = generate_tokens_paddle(request.inputFilePath)
        prompt_val = get_file_content(request.promptFilePath)
        messages = [
            {"role": "system", "content": prompt_val},
            {"role": "user", "content": ocr_result},
        ]

        # prompt = pipeline.tokenizer.apply_chat_template(
        #     messages,
        #     tokenize=False,
        #     add_generation_prompt=True,
        # )
        #
        # terminators = [
        #     pipeline.tokenizer.eos_token_id,
        #     pipeline.tokenizer.convert_tokens_to_ids("<|eot_id|>")
        # ]

        prompt = '[INST] Hi there, write me 3 random quotes [/INST]'

        stream = pipeline(
            prompt,  # Prompt
            max_tokens=4046,  # Generate up to 512 tokens
            stop=["</s>"],
            # Example stop token - not necessarily correct for this specific model! Please check before using.
            echo=False  # Whether to echo the prompt
        )
        result = ""
        for output in stream:
            result += output['choices'][0]['text']
            print(output['choices'][0]['text'], end="")

        # prompt_result = outputs[0]["generated_text"][len(prompt):]
        prompt_result = result
        return prompt_result

    except Exception as ex:
        raise ex
    finally:
        gc.collect()
        torch.cuda.empty_cache()


@app.post("/argon/text")
def text_extraction_by_paddle(image_path: str):
    return generate_tokens_paddle(image_path)
