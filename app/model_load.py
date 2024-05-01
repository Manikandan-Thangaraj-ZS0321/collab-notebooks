import torch
import logging
import transformers

from unsloth import FastLanguageModel
from paddleocr import PaddleOCR

model_id = "/home/hera/workspace/llama3/models--meta-llama--Meta-Llama-3-8B-Instruct/snapshots/e5e23bbe8e749ef0efcf16cad411a7d23bd23298"


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
            )
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
                model_name="unsloth/llama-3-8b-Instruct",
                max_seq_length=max_seq_length,
                dtype=dtype,
                load_in_4bit=load_in_4bit
            )
            logging.info("The krypton chat model has been successfully loaded")
            return model, tokenizer
        except Exception as ex:
            logging.error("Error in loading krypton chat model")
            raise ex

    @staticmethod
    def paddleocr_model_load():
        return PaddleOCR(use_angle_cls=True, lang='en', enable_mkldnn=False, show_log=False)
