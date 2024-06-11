import torch
import streamlit as st
import gc
import os
import PyPDF2
import uuid

from PIL import Image
from transformers import AutoModelForCausalLM, AutoTokenizer
from pdf2image import convert_from_path


# Constants and Model setup
MODEL_PATH = "THUDM/cogvlm2-llama3-chat-19B"
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
TORCH_TYPE = torch.bfloat16 if torch.cuda.is_available() and torch.cuda.get_device_capability()[0] >= 8 else torch.float16

tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH, trust_remote_code=True)


@st.cache_resource
def get_model_load():
    extraction_model = AutoModelForCausalLM.from_pretrained(MODEL_PATH, torch_dtype=TORCH_TYPE, trust_remote_code=True,low_cpu_mem_usage=True).eval()
    return extraction_model


model = get_model_load()

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


def generate_response(query, image=None):
    if image is None:
        query = text_only_template.format(query)
    
    inputs = model.build_conversation_input_ids(
        tokenizer,
        query=query,
        history=None,
        images=[image] if image else None,
        template_version='chat'
    )
    
    input_data = {
        'input_ids': inputs['input_ids'].unsqueeze(0).to(DEVICE),
        'token_type_ids': inputs['token_type_ids'].unsqueeze(0).to(DEVICE),
        'attention_mask': inputs['attention_mask'].unsqueeze(0).to(DEVICE),
        'images': [[inputs['images'][0].to(DEVICE).to(TORCH_TYPE)]] if image is not None else None,
    }
    
    gen_kwargs = {
        "max_new_tokens": 8192,
        "pad_token_id":  128002
    }
    
    with torch.no_grad():
        outputs = model.generate(**input_data, **gen_kwargs)
        outputs = outputs[:, input_data['input_ids'].shape[1]:]
        model_response = tokenizer.decode(outputs[0], skip_special_tokens=True)
    
    # Clear GPU memory
    torch.cuda.empty_cache()
    del input_data, outputs
    gc.collect()
    
    return model_response


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
                        image = Image.open(image_file).convert('RGB')
                    else:
                        image = None

                    response = generate_response(query, image)
                    with st.chat_message("assistant"):
                        st.markdown(response)

                    st.session_state.messages.append({"role": "assistant", "content": response})

        if file_type in ("image/jpg", "image/jpeg", "image/png"):

            with st.spinner('Processing...'):
                if uploaded_file is not None:
                    image = Image.open(uploaded_file).convert('RGB')
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
