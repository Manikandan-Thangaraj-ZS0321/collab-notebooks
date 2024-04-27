from fastapi import FastAPI
import transformers
import torch
import gc
from paddleocr import PaddleOCR

model_id = "meta-llama/Meta-Llama-3-8B-Instruct"

# pipeline = transformers.pipeline(
#     "text-generation",
#     model=model_id,
#     model_kwargs={
#         "torch_dtype": torch.float16,
#         "quantization_config": {"load_in_4bit": True},
#         "low_cpu_mem_usage": True,
#     },
# )

pipeline = transformers.pipeline(
    "text-generation",
    model="meta-llama/Meta-Llama-3-8B-Instruct",
    model_kwargs={"torch_dtype": torch.bfloat16},
    device="cuda",
)

app = FastAPI()
paddle_ocr = PaddleOCR(use_angle_cls=True, lang='en', enable_mkldnn=False)


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
def read_item(inputFilePath: str, prompt: str):
    try:
        ocr_result = generate_tokens_paddle_(inputFilePath)
        messages = [
            {"role": "system", "content": prompt},
            {"role": "user", "content": ocr_result},
        ]

        prompt = pipeline.tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
            format="JSON"
        )

        terminators = [
            pipeline.tokenizer.eos_token_id,
            pipeline.tokenizer.convert_tokens_to_ids("<|eot_id|>")
        ]

        outputs = pipeline(
            prompt,
            max_new_tokens=2048,
            eos_token_id=terminators,
            do_sample=True,
            temperature=0.6,
            top_p=0.9,
        )
        prompt_result = outputs[0]["generated_text"][len(prompt):]
        print(prompt_result)
        return prompt_result

    except Exception as ex:
        raise ex
    finally:
        gc.collect()
        torch.cuda.empty_cache()
