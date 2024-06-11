import requests
import torch
from PIL import Image
from io import BytesIO
import json
import os
from datetime import datetime, time
from fastapi import FastAPI
import cv2
from PIL import Image

app = FastAPI()


from transformers import AutoProcessor, AutoModelForVision2Seq
from transformers.image_utils import load_image
from transformers import BitsAndBytesConfig

DEVICE = "cuda:0"

quantization_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_use_double_quant=True,
    bnb_4bit_compute_dtype=torch.float16
)

processor = AutoProcessor.from_pretrained("HuggingFaceM4/idefics2-8b-chatty", do_image_splitting=False)
model = AutoModelForVision2Seq.from_pretrained(
    "HuggingFaceM4/idefics2-8b-chatty",
    torch_dtype=torch.float16,
    quantization_config=quantization_config,
    low_cpu_mem_usage=True
)

#image1 = load_image("/home/falcon/intics-build/data/data/output/pdf_to_image/SYNT_167074460/SYNT_167074460_0.jpg")
image1 = load_image("/home/falcon/intics-build/data/data/output/idefics2_results/agadia_samples/SYNT_166522063_1.jpg")
image2 = load_image("/home/falcon/intics-build/data/data/output/idefics2_results/agadia_samples/SYNT_166529664_1.jpg")
# image1 = load_image("/home/falcon/intics-build/data/data/output/idefics2_results/agadia_samples/")
# image1 = load_image("/home/falcon/intics-build/data/data/output/idefics2_results/agadia_samples/")
# image1 = load_image("/home/falcon/intics-build/data/data/output/idefics2_results/agadia_samples/")
# image1 = load_image("/home/falcon/intics-build/data/data/output/idefics2_results/agadia_samples/")

#image1 = load_image("/home/falcon/intics-build/data/data/output/Anthem/HandWritten1/HandWritten1_2.jpg")


@app.post("/idefics_chatty")
def process_files_in_directory(filePath: str):

    start_time = datetime.now().time()

    prompt_val = get_file_content("response_prompt_v2.txt")
    messages = [
        {
            "role": "user",
            "content": [
                {"type": "image"},
                {"type": "text",
                 "text": "Extract all key-value pairs from the document and provide the results in a JSON format."},
            ]
        },
    ]

    prompt = processor.apply_chat_template(messages, add_generation_prompt=True)

    image = Image.open(filePath)

    target_size = (980, 980)  # Set the desired width and height
    resized_image = image.resize(target_size)

    inputs = processor(text=prompt, images=[resized_image], return_tensors="pt")
    inputs = {k: v.to(DEVICE) for k, v in inputs.items()}

    # Generate
    generated_ids = model.generate(**inputs, max_new_tokens=8192)
    generated_texts = processor.batch_decode(generated_ids, skip_special_tokens=True)

    end_time = datetime.now().time()
    time_diff = datetime.combine(datetime.today(), start_time) - datetime.combine(datetime.today(), end_time)
    time_diff_seconds = time_diff.total_seconds()
    print("Time difference in seconds:", time_diff_seconds)

    print(generated_texts)
    return generated_texts


def get_file_content(file_path):
    with open(file_path, 'r') as file:
        text = file.read()
        return text

# Create inputs




# ['User: What do we see in this image? \nAssistant: In this image, we can see the city of New York, and more specifically the Statue of Liberty. \nUser: And how about this image? \nAssistant: In this image we can see buildings, trees, lights, water and sky.']

