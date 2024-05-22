import requests
import torch
from PIL import Image
from io import BytesIO

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

processor = AutoProcessor.from_pretrained("HuggingFaceM4/idefics2-8b", do_image_splitting=False)
model = AutoModelForVision2Seq.from_pretrained(
    "HuggingFaceM4/idefics2-8b",
    torch_dtype=torch.float16,
    quantization_config=quantization_config,
)

def get_file_content(file_path):
    with open(file_path, 'r') as file:
        text = file.read()
        return text

image1 = load_image("/home/hera/Downloads/test_folder/SYNT_166564144_1.jpg")
image2 = load_image("https://cdn.britannica.com/59/94459-050-DBA42467/Skyline-Chicago.jpg")
image3 = load_image("https://cdn.britannica.com/68/170868-050-8DDE8263/Golden-Gate-Bridge-San-Francisco.jpg")

prompt_val = get_file_content("response_prompt_v2.txt")
# Create inputs
messages = [
    {
        "role": "user",
        "content": [
            {"type": "image"},
            {"type": "text", "text": "Extract all the key value pairs"},
        ]
    },
]


prompt = processor.apply_chat_template(messages, add_generation_prompt=True)
inputs = processor(text=prompt, images=[image1], return_tensors="pt")
inputs = {k: v.to(DEVICE) for k, v in inputs.items()}


# Generate
generated_ids = model.generate(**inputs, max_new_tokens=8192)
generated_texts = processor.batch_decode(generated_ids, skip_special_tokens=True)

print(generated_texts)
# ['User: What do we see in this image? \nAssistant: In this image, we can see the city of New York, and more specifically the Statue of Liberty. \nUser: And how about this image? \nAssistant: In this image we can see buildings, trees, lights, water and sky.']
