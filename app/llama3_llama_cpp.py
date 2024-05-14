import torch
import gc
import os
import json

from fastapi import FastAPI, File, UploadFile, Query
from fastapi.responses import StreamingResponse

from pydantic import BaseModel
from fastapi import FastAPI
from model_load import ModelLoad
from text_extraction import TextExtraction
from logger_handler import logger
from typing import List

pipeline = ModelLoad.krypton_chat_llamacpp_model_load()
text_argon_model = TextExtraction.argon_text_model_load()
text_xenon_model = TextExtraction.xenon_text_model_load()
processor, text_krypton_model = TextExtraction.krypton_text_model_load()


app = FastAPI()


class ApiRequest(BaseModel):
    files: List[str]
    outputFolder: str
    textExtractionModel: str


class LlamaRequest(BaseModel):
    inputFilePath: str
    promptFilePath: str
    textExtractionModel: str


class ChatRequest(BaseModel):
    inputFilePath: str
    textExtractionModel: str


def get_file_content(file_path):
    with open(file_path, 'r') as file:
        text = file.read()
        return text


@app.post("/chat/llama/files")
def chat_with_files(request: ChatRequest, prompt: str = Query(...)):
    try:
        text_extraction_model = request.textExtractionModel

        if text_extraction_model == "ARGON":
            ocr_result = TextExtraction.text_extraction_argon(request.inputFilePath, text_argon_model)
        elif text_extraction_model == "KRYPTON":
            ocr_result = TextExtraction.text_extraction_krypton(request.inputFilePath, processor, text_krypton_model)
        else:
            ocr_result = TextExtraction.text_extraction_xenon(request.inputFilePath, text_xenon_model)

        prompt_val = prompt

        messages = [
            {"role": "system", "content": prompt_val},
            {"role": "user", "content": ocr_result},
        ]

        # response = pipeline.create_chat_completion(messages=messages, response_format={"type": "json_object"})
        response = pipeline.create_chat_completion(messages=messages, stream=True)

        prompt_result = response["choices"][0]["message"]["content"].strip()
        # prompt_result = response

        logger.info("completed response generation from llm")

        return response

    except Exception as ex:
        raise ex
    finally:
        gc.collect()
        torch.cuda.empty_cache()


@app.post("/chat/llama")
def chat_prompt(prompt: str = Query(...)):
    try:
        prompt_val = prompt

        messages = [
            {"role": "system", "content": prompt_val},
            {"role": "user", "content": ""},
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


def process_file(request: LlamaRequest):
    try:
        text_extraction_model = request.textExtractionModel

        if text_extraction_model == "ARGON":
            ocr_result = TextExtraction.text_extraction_argon(request.inputFilePath, text_argon_model)
        elif text_extraction_model == "KRYPTON":
            ocr_result = TextExtraction.text_extraction_krypton(request.inputFilePath, processor, text_krypton_model)
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
    return TextExtraction.text_extraction_xenon(image_path, text_xenon_model)


@app.post("/krypton/text")
def text_extraction_by_krypton(image_path: str):
    return TextExtraction.text_extraction_krypton(image_path, processor, text_krypton_model)


@app.post("/multipart-upload")
async def create_papers(file: UploadFile = File(...)):
    try:
        outputDir = "/home/hera/data/output"
        if not os.path.exists(outputDir):
            os.makedirs(outputDir, exist_ok=True)

        output_file = os.path.join(outputDir, file.filename)

        if os.path.exists(output_file):
            logger.info("File {} already exists".format(file.filename))
            return {"message": "File already downloaded", "filename": file.filename}

        with open(output_file, "wb") as f:
            f.write(file.file.read())

        logger.info("File {} downloaded successfully".format(file.filename))
        return output_file

    except Exception as e:
        logger.error("Error in downloading multipart file {} with exception {}".format(file.filename, e))
        return {"error": str(e)}
