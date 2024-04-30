from fastapi import FastAPI
import torch
import gc
from paddleocr import PaddleOCR
from pydantic import BaseModel
from unsloth import FastLanguageModel
from transformers import TextStreamer

model_id = "unsloth/Meta-Llama-3-8B-Instruct"

max_seq_length = 2048
dtype = None  # None for auto detection. Float16 for Tesla T4, V100, Bfloat16 for Ampere+
load_in_4bit = True

model, tokenizer = FastLanguageModel.from_pretrained(
    model_name="unsloth/llama-3-8b-Instruct",
    max_seq_length=max_seq_length,
    dtype=dtype,
)

alpaca_prompt = """Below is an instruction that describes a task, paired with an input that provides further context. Write a response that appropriately completes the request.

Instruction:
{}

Input:
{}

Response:
{}"""

FastLanguageModel.for_inference(model)

app = FastAPI()
paddle_ocr = PaddleOCR(use_angle_cls=True, lang='en', enable_mkldnn=False, show_log=False)


class LlamaRequest(BaseModel):
    inputFilePath: str
    promptFilePath: str


def get_file_content(file_path):
    with open(file_path, 'r') as file:
        text = file.read()
        return text


def generate_tokens_paddle_(image_path: str) -> str:
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


@app.post("/llama3")
def read_item(request: LlamaRequest):
    try:
        ocr_result = generate_tokens_paddle_(request.inputFilePath)
        prompt_val = get_file_content(request.promptFilePath)

        inputs = tokenizer(
            [
                alpaca_prompt.format(
                    prompt_val,  # instruction
                    ocr_result,  # input
                    "",  # output - leave this blank for generation!
                )
            ], return_tensors="pt").to("cuda")

        text_streamer = TextStreamer(tokenizer)
        outputs = model.generate(**inputs, streamer=text_streamer, max_new_tokens=2048)
        generated_responses = tokenizer.batch_decode(outputs)

        response = llm_post_processing_latest(generated_responses)
        return response

    except Exception as ex:
        raise ex
    finally:
        gc.collect()
        torch.cuda.empty_cache()


def llm_post_processing_latest(generated_responses: str):
    if generated_responses is not None:
        for generated_response in generated_responses:
            start_index = generated_response.find("Response:") + len("Response:")
            end_index = generated_response.find("</s>", start_index)
            response_section = generated_response[start_index:end_index].strip()
            return response_section
    else:
        return []
