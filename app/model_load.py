import torch
import transformers
import os

from unsloth import FastLanguageModel
from paddleocr import PaddleOCR
from llama_cpp import Llama
from logger_handler import logger

HF_TOKEN = os.environ['HF_TOKEN']
#model_id = "/home/hera/workspace/llama3/models--meta-llama--Meta-Llama-3-8B-Instruct/snapshots/e5e23bbe8e749ef0efcf16cad411a7d23bd23298"
#unsloth_model = "/home/hera/workspace/llama3/models--meta-llama--Meta-Llama-3-8B-Instruct/snapshots/e5e23bbe8e749ef0efcf16cad411a7d23bd23298"

# model_id= "/data/models/models--meta-llama--Meta-Llama-3-8B-Instruct/snapshots/e5e23bbe8e749ef0efcf16cad411a7d23bd23298"
# unsloth_model = "/data/models/models--unsloth--llama-3-8b-Instruct-bnb-4bit/snapshots/efa44c86af4fcbbc3d75e6cb1c8bfaf7f5c7cfc1"

model_id = "meta-llama/Meta-Llama-3-8B-Instruct"
unsloth_model = "unsloth/llama-3-8b-Instruct"
llama_cpp_model = "QuantFactory/Meta-Llama-3-8B-Instruct-GGUF"


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
            logger.info("The krypton chat model has been successfully loaded")
            return pipeline
        except Exception as ex:
            logger.error("Error in loading krypton chat model")
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
                token=HF_TOKEN
            )
            logger.info("The krypton chat model has been successfully loaded")
            return model, tokenizer
        except Exception as ex:
            logger.error("Error in loading krypton chat model")
            raise ex

    @staticmethod
    def krypton_chat_llamacpp_model_load():

        try:
            max_seq_length = 2048
            dtype = None
            load_in_4bit = True
            llm = Llama(
                model_path=llama_cpp_model,  # Download the model file first
                n_ctx=max_seq_length,
                # The max sequence length to use - note that longer sequence lengths require much more resources
                n_threads=8,  # The number of CPU threads to use, tailor to your system and the resulting performance
                n_gpu_layers=1  # The number of layers to offload to GPU, if you have GPU acceleration available
            )
            logger.info("The krypton chat model has been successfully loaded")
            return llm
        except Exception as ex:
            logger.error("Error in loading krypton chat model")
            raise ex

    @staticmethod
    def paddleocr_model_load():
        return PaddleOCR(use_angle_cls=True, lang='en', show_log=False, use_gpu=True)
