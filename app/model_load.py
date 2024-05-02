import torch
import logging
import transformers
import os

from unsloth import FastLanguageModel
from paddleocr import PaddleOCR

HF_TOKEN = os.environ['HF_TOKEN']
#model_id = "/home/hera/workspace/llama3/models--meta-llama--Meta-Llama-3-8B-Instruct/snapshots/e5e23bbe8e749ef0efcf16cad411a7d23bd23298"
#unsloth_model = "/home/hera/workspace/llama3/models--meta-llama--Meta-Llama-3-8B-Instruct/snapshots/e5e23bbe8e749ef0efcf16cad411a7d23bd23298"

# model_id= "/data/models/models--meta-llama--Meta-Llama-3-8B-Instruct/snapshots/e5e23bbe8e749ef0efcf16cad411a7d23bd23298"
# unsloth_model = "/data/models/models--unsloth--llama-3-8b-Instruct-bnb-4bit/snapshots/efa44c86af4fcbbc3d75e6cb1c8bfaf7f5c7cfc1"

model_id= "meta-llama/Meta-Llama-3-8B-Instruct"
unsloth_model = "unsloth/llama-3-8b-Instruct"

class ModelLoad:
    def __init__(self):
        pass

    @staticmethod
    def krypton_chat_model_load():
        try:
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
                model=model_id,
                model_kwargs={"torch_dtype": torch.bfloat16},
                device="cuda",
                token=HF_TOKEN
            )
            logging.info("The krypton chat model has been successfully loaded")
            return pipeline
        except Exception as ex:
            logging.error("Error in loading krypton chat model")
            raise ex

    @staticmethod
    def krypton_chat_4bit_model_load():

        try:
            max_seq_length = 2048
            dtype = None
            load_in_4bit = True
            model, tokenizer = FastLanguageModel.from_pretrained(
                model_name=unsloth_model,
                max_seq_length=max_seq_length,
                dtype=dtype,
                load_in_4bit=load_in_4bit,
                use_auth_token=HF_TOKEN
            )
            logging.info("The krypton chat model has been successfully loaded")
            return model, tokenizer
        except Exception as ex:
            logging.error("Error in loading krypton chat model")
            raise ex

    @staticmethod
    def paddleocr_model_load():
        return PaddleOCR(use_angle_cls=True, lang='en', enable_mkldnn=False, show_log=False)
