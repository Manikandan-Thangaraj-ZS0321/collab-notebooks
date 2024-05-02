import torch
import gc
import os
import json
import time

from pydantic import BaseModel
from fastapi import FastAPI
from model_load import ModelLoad
from transformers import TextStreamer
from typing import List

model, tokenizer = ModelLoad.krypton_chat_4bit_model_load()
app = FastAPI()
paddle_ocr = ModelLoad.paddleocr_model_load()

alpaca_prompt = """Below is an instruction that describes a task, paired with an input that provides further context. Write a response that appropriately completes the request.

Instruction:
{}

Input:
{}

Response:
{}"""


class ApiRequest(BaseModel):
    files: List[str]
    outputFolder: str


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


def process_file(request: LlamaRequest):
    try:
        ocr_result = generate_tokens_paddle(request.inputFilePath)
        prompt_val = get_file_content(request.promptFilePath)

        inputs = tokenizer(
            [
                alpaca_prompt.format(
                    prompt_val,  # instruction
                    ocr_result,  # input
                    "",  # output - leave this blank for generation!
                )
            ], return_tensors="pt").to("cuda")

        # use a TextStreamer for continuous inference - so you can see the generation token by token
        # text_streamer = TextStreamer(tokenizer)
        # outputs = model.generate(**inputs, streamer=text_streamer, max_new_tokens=2048, temperature=0.1)

        outputs = model.generate(**inputs, max_new_tokens=2048, use_cache=True, temperature=0.1)
        generated_responses = tokenizer.batch_decode(outputs, skip_special_tokens=True)

        response = llm_post_processing_latest(generated_responses)
        return response

    except Exception as ex:
        raise ex
    finally:
        gc.collect()
        torch.cuda.empty_cache()


def get_json_data(text):
    start_index = text.find('{')
    end_index = text.rfind('}')
    json_content = text[start_index:end_index + 1]
    try:
        json_data = json.loads(json_content)
        return json_data
    except json.JSONDecodeError as e:
        return json_content


def llm_post_processing_latest(generated_responses: str):
    if generated_responses is not None:
        for generated_response in generated_responses:
            start_index = generated_response.find("Response:") + len("Response:")
            end_index = generated_response.find("</s>", start_index)
            response_section = generated_response[start_index:end_index].strip()
            return response_section
    else:
        return []


@app.post("/argon/text")
def text_extraction_by_paddle(image_path: str):
    return generate_tokens_paddle(image_path)


@app.post("/chat/krypton/files")
def process_files_in_directory(request: ApiRequest):
    files = request.files
    output_folder = request.outputFolder

    for file in files:

        if output_folder == "":
            json_file_path = os.path.splitext(file)[0] + ".json"
        else:
            if not os.path.exists(output_folder):
                os.makedirs(output_folder, exist_ok=True)

            filename = os.path.basename(file)
            json_file_path = os.path.join(output_folder, f"{filename}.json")

        # Process the image
        llama_request = LlamaRequest(inputFilePath=file, promptFilePath="response_prompt_v2.txt")
        llama_response = process_file(llama_request)

        json_response = get_json_data(llama_response)

        with open(json_file_path, "w") as json_file:
            json.dump(json_response, json_file, indent=4)  # Format JSON with indentation

        return json_response
