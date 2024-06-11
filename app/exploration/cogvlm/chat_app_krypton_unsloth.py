import torch
import streamlit as st
import gc
import os
import PyPDF2
import uuid
import transformers

import ssl

ssl._create_default_https_context = ssl._create_unverified_context

from transformers import AutoTokenizer
from pdf2image import convert_from_path
from doctr.models import ocr_predictor
from doctr.io import DocumentFile
from unsloth import FastLanguageModel


max_seq_length = 2048
dtype = None
load_in_4bit = True
# Constants and Model setup
unsloth_model = "/home/hera/workspace/llama3/unsloth/models--unsloth--llama-3-8b-Instruct-bnb-4bit/snapshots/efa44c86af4fcbbc3d75e6cb1c8bfaf7f5c7cfc1"
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
TORCH_TYPE = torch.bfloat16 if torch.cuda.is_available() and torch.cuda.get_device_capability()[0] >= 8 else torch.float16


@st.cache_resource
def get_model_load():
    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=unsloth_model,
        max_seq_length=max_seq_length,
        dtype=dtype,
        load_in_4bit=load_in_4bit,
    )

    return model, tokenizer


@st.cache_resource
def xenon_text_model_load():
    if torch.cuda.is_available():
        return ocr_predictor(pretrained=True).cuda()
    else:
        return ocr_predictor(pretrained=True).cpu()


pipeline, tokenizer = get_model_load()
text_xenon_model = xenon_text_model_load()

text_only_template = "A chat between a curious user and an artificial intelligence assistant. The assistant gives helpful, detailed, and polite answers to the user's questions. USER: {} ASSISTANT:"

# Emojis for user and assistant
USER_EMOJI = "🧑"
ASSISTANT_EMOJI = "🤖"

# Title
st.title("How Can I Help You?")


def save_uploaded_file(uploaded_file, save_dir):
    os.makedirs(save_dir, exist_ok=True)
    file_path = os.path.join(save_dir, uploaded_file.name)
    with open(file_path, "wb") as f:
        f.write(uploaded_file.getbuffer())
    return file_path


def get_num_pages(pdf_file):
    pdf_reader = PyPDF2.PdfReader(pdf_file)
    return len(pdf_reader.pages)


with st.sidebar:
    st.image("https://demo.intics.ai/intics_logo_blue.png")
    st.title("Chatbot")
    uploaded_file = st.file_uploader("Choose file", accept_multiple_files=False, type=["pdf", "jpg", "jpeg", "png"])

    if uploaded_file:
        file_type = uploaded_file.type
        if file_type == 'application/pdf':
            # pdf uploader
            save_dir = "data/pdf_uploader"
            saved_file_path = save_uploaded_file(uploaded_file, save_dir)
            num_pages = get_num_pages(uploaded_file)
            selected_page = st.number_input("pdf page", value=1, min_value = 1, max_value= num_pages, step=1, format="%d", key="number_input")


if "messages" not in st.session_state:
    st.session_state.messages = []

for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])


def text_extraction_xenon(file_path, model):
    try:
        doc = DocumentFile.from_images(file_path)
        output = model(doc)
        json_output = output.export()
        # words_with_coordinates = get_word_coordinates(json_output)
        words = get_words(json_output)
        paragraph = ' '.join(words)
        return paragraph
    except Exception as ex:
        raise ex
    finally:
        gc.collect()
        torch.cuda.empty_cache()

def get_words(output):
    try:
        # page_dim = output['pages'][0]["dimensions"]
        text_coordinates = []
        for obj1 in output['pages'][0]["blocks"]:
            for obj2 in obj1["lines"]:
                for obj3 in obj2["words"]:
                    text_coordinates.append(obj3["value"])
        return text_coordinates
    except Exception as ex:
        raise ex


def generate_response(query, image=None):
    if image is None:
        prompt_val = text_only_template.format(query)
        messages = [
            {"role": "system", "content": prompt_val},
            {"role": "user", "content": ""},
        ]
    else:
        prompt_val = query
        ocr_result = text_extraction_xenon(image, text_xenon_model)
        messages = [
            {"role": "system", "content": prompt_val},
            {"role": "user", "content": ocr_result},
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

    outputs = pipeline.generate(input_ids=prompt, max_new_tokens=8192, use_cache=True,
                             pad_token_id=tokenizer.pad_token_id)

    generated_responses = tokenizer.batch_decode(outputs, skip_special_tokens=True)

    assistant_response = generated_responses[0]

    # Clear GPU memory
    torch.cuda.empty_cache()
    del outputs
    gc.collect()
    
    return assistant_response


def split_pdf(pdf_path, output_path):
    with open(pdf_path, 'rb') as file:
        reader = PyPDF2.PdfReader(file)
        num_pages = reader.numPages
        for page_number in range(num_pages):
            writer = PyPDF2.PdfFileWriter()
            writer.addPage(reader.getPage(page_number))
            output_file_path = os.path.join(output_path, f"{page_number + 1}.pdf")
            with open(output_file_path, 'wb') as output_file:
                writer.write(output_file)
    return output_path


def pdf_to_images(pdf_path, output_folder, image_format='JPEG', resolution=300):
    os.makedirs(output_folder, exist_ok=True)
    images = convert_from_path(pdf_path, dpi=resolution)
    image_output_paths = []
    for i, image in enumerate(images):
        image_path = os.path.join(output_folder, f"page_{i + 1}.{image_format.lower()}")
        image.save(image_path, image_format)
        image_output_paths.append(image_path)
    return image_output_paths


query = st.chat_input("Enter Prompt:")
if query:

    with st.chat_message("user"):
        st.markdown(query)

    st.session_state.messages.append({"role": "user", "content": query})

    if uploaded_file is not None:

        file_type = uploaded_file.type

        if file_type == 'application/pdf':

            with st.spinner('Processing...'):

                filename = os.path.splitext(os.path.basename(saved_file_path))[0]

                pdf_images_base_path = f"data/image_uploader/{filename}"
                image_paths = pdf_to_images(saved_file_path, pdf_images_base_path)

                if 1 <= selected_page <= len(image_paths):
                    image_file = image_paths[selected_page - 1]
                    if image_file is not None:
                        image = image_file
                    else:
                        image = None

                    response = generate_response(query, image)
                    with st.chat_message("assistant"):
                        st.markdown(response)

                    st.session_state.messages.append({"role": "assistant", "content": response})

        if file_type in ("image/jpg", "image/jpeg", "image/png"):

            with st.spinner('Processing...'):
                if uploaded_file is not None:
                    save_dir = "data/image_uploader"
                    saved_image_path = save_uploaded_file(uploaded_file, save_dir)
                    image = saved_image_path
                else:
                    image = None
                response = generate_response(query, image)
                with st.chat_message("assistant"):
                    st.markdown(response)

                st.session_state.messages.append({"role": "assistant", "content": response})

    else:
        with st.spinner('Processing...'):
            response = generate_response(query)
            with st.chat_message("assistant"):
                st.markdown(response)

            st.session_state.messages.append({"role": "assistant", "content": response})
