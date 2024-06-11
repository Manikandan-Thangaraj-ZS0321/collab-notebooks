import torch
from PIL import Image
from transformers import AutoModelForCausalLM, AutoTokenizer
import streamlit as st
import gc

# Constants and Model setup
MODEL_PATH = "THUDM/cogvlm2-llama3-chat-19B-int4"
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

# Streamlit UI
st.title("How Can I Help You?")

# Sidebar for file upload
st.sidebar.title("InticsGPT 2.0")

uploaded_file = st.sidebar.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])
if uploaded_file is not None:
    image = Image.open(uploaded_file).convert('RGB')
    #st.image(image, caption='Uploaded Image.', use_column_width=True)
else:
    image = None

# Chat interaction
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
        "max_new_tokens": 2048,
        "pad_token_id":  128002      #  tokenizer.pad_token_id,  # Adjust based on the tokenizer's pad token id
    }
    
    with torch.no_grad():
        outputs = model.generate(**input_data, **gen_kwargs)
        outputs = outputs[:, input_data['input_ids'].shape[1]:]
        response = tokenizer.decode(outputs[0], skip_special_tokens=True)
    
    # Clear GPU memory
    torch.cuda.empty_cache()
    del input_data, outputs
    gc.collect()
    
    return response

query = st.chat_input("Enter Prompt:")
if query:
    st.write(f"{USER_EMOJI} **You**: {query}")
    response = generate_response(query, image)
    #st.write(f"{ASSISTANT_EMOJI} **Assistant**: {response}")
    st.write(response)

