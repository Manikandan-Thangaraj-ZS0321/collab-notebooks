import torch
import gc
import os
import json

from pydantic import BaseModel
from fastapi import FastAPI
from model_load import ModelLoad
from text_extraction import TextExtraction
from logger_handler import logger
from typing import List

pipeline = ModelLoad.krypton_chat_llamacpp_model_load()
text_argon_model = TextExtraction.argon_text_model_load()
text_xenon_model = TextExtraction.xenon_text_model_load()

app = FastAPI()


class ApiRequest(BaseModel):
    files: List[str]
    outputFolder: str
    textExtractionModel: str


class LlamaRequest(BaseModel):
    inputFilePath: str
    promptFilePath: str
    textExtractionModel: str


def get_file_content(file_path):
    with open(file_path, 'r') as file:
        text = file.read()
        return text


@app.post("/chat/krypton")
def process_file(request: LlamaRequest):
    try:
        text_extraction_model = request.textExtractionModel

        if text_extraction_model == "ARGON":
            ocr_result = TextExtraction.text_extraction_argon(request.inputFilePath, text_argon_model)
        else:
            ocr_result = TextExtraction.text_extraction_xenon(request.inputFilePath, text_xenon_model)

        prompt_val = get_file_content(request.promptFilePath)

        messages = [
            {"role": "system", "content": prompt_val},
            {"role": "user", "content": ocr_result},
        ]

        # response = pipeline.create_chat_completion(messages=messages, response_format={"type": "json_object"})
        response = pipeline.create_chat_completion(messages=messages)

        prompt_result = response["choices"][0]["message"]["content"].strip()
        # prompt_result = response

        logger.info("completed response generation from llm")

        return prompt_result

    except Exception as ex:
        raise ex
    finally:
        gc.collect()
        torch.cuda.empty_cache()


@app.post("/chat/krypton/files")
def process_files_in_directory(request: ApiRequest):
    files = request.files
    output_folder = request.outputFolder

    json_response = ""
    for file in files:

        if output_folder == "":
            json_file_path = os.path.splitext(file)[0] + ".json"
        else:
            if not os.path.exists(output_folder):
                os.makedirs(output_folder, exist_ok=True)

            filename = os.path.splitext(os.path.basename(file))[0]
            json_file_path = os.path.join(output_folder, f"{filename}.json")

        # Process the image
        llama_request = LlamaRequest(inputFilePath=file, promptFilePath="prompts/response_prompt_v2.txt", textExtractionModel=request.textExtractionModel)
        llama_response = process_file(llama_request)

        json_response = get_json_data(llama_response)

        with open(json_file_path, "w") as json_file:
            json.dump(json_response, json_file, indent=4)  # Format JSON with indentation

    return json_response


def get_json_data(text):
    start_index = text.find('{')
    end_index = text.rfind('}')
    json_content = text[start_index:end_index + 1]
    try:
        json_data = json.loads(json_content)
        logger.info("successful in loading data as json")
        return json_data
    except json.JSONDecodeError as e:
        logger.error("Error in converting data to json format with exception {}".format(e))
        return json_content


@app.post("/argon/text")
def text_extraction_by_paddle(image_path: str):
    return TextExtraction.text_extraction_argon(image_path, text_argon_model)


@app.post("/xenon/text")
def text_extraction_by_xenon(image_path: str):
    return TextExtraction.text_extraction_argon(image_path, text_xenon_model)
