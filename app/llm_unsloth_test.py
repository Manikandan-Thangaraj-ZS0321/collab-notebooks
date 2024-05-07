import torch
import gc
import os
import json

from pydantic import BaseModel
from fastapi import FastAPI
from model_load import ModelLoad
from transformers import TextStreamer
from typing import List
#from outlines import generate
from logger_handler import logger
from text_extraction import TextExtraction


model, tokenizer = ModelLoad.krypton_chat_4bit_model_load()
pipeline = ModelLoad.krypton_chat_llamacpp_model_load()
text_argon_model = TextExtraction.argon_text_model_load()
text_xenon_model = TextExtraction.xenon_text_model_load()


app = FastAPI()
EOS_TOKEN = tokenizer.eos_token


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


def process_file(request: LlamaRequest):
    filename = os.path.basename(request.inputFilePath)
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
            {"role": "assistant:", "content": ""},
        ]

        terminators = [
            tokenizer.eos_token_id,
            tokenizer.convert_tokens_to_ids("<|eot_id|>")
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

        # use a TextStreamer for continuous inference - so you can see the generation token by token
        # text_streamer = TextStreamer(tokenizer)
        # outputs = model.generate(input_ids=prompt, streamer=text_streamer, max_new_tokens=2048, use_cache=True,
        #                                  pad_token_id=tokenizer.pad_token_id)

        outputs = model.generate(input_ids=prompt, max_new_tokens=2048, use_cache=True,
                                 pad_token_id=tokenizer.pad_token_id)

        logger.info("completed response generation for file {}".format(filename))
        generated_responses = tokenizer.batch_decode(outputs, skip_special_tokens=True)

        assistant_response = generated_responses[0]

        response = llm_post_processing_latest(assistant_response)

        return response

    except Exception as ex:
        logger.error("Error in getting results for file {} with exception {}".format(filename, ex))
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
        logger.info("successful in loading data as json")
        return json_data
    except json.JSONDecodeError as e:
        logger.error("Error in converting data to json format with exception {}".format(e))
        return json_content


def llm_post_processing_latest(generated_response: str):
    if generated_response is not None:
        start_index = generated_response.find("assistant") + len("assistant")
        end_index = generated_response.find("</s>", start_index)
        response_section = generated_response[start_index:end_index].strip()
        logger.info("completed the post processing of llm response")
        return response_section
    else:
        logger.error("Error in post processing results - generated response is none")
        return []


@app.post("/argon/text")
def text_extraction_by_paddle(image_path: str):
    return TextExtraction.text_extraction_argon(image_path, text_argon_model)


@app.post("/xenon/text")
def text_extraction_by_xenon(image_path: str):
    return TextExtraction.text_extraction_argon(image_path, text_xenon_model)


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
