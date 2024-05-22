import torch
from PIL import Image
from transformers import AutoModelForCausalLM, AutoTokenizer

model_id = "qresearch/llama-3-vision-alpha-hf"
model = AutoModelForCausalLM.from_pretrained(
    model_id, trust_remote_code=True, torch_dtype=torch.float16
).to("cuda")

tokenizer = AutoTokenizer.from_pretrained(
    model_id,
    use_fast=True,
)

image = Image.open("image_path")

print(
    tokenizer.decode(
        model.answer_question(image, "question", tokenizer),
        skip_special_tokens=True,
    )
)
