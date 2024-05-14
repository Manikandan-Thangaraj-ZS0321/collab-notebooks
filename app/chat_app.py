import streamlit as st
import requests
from llama_cpp_model_load import ModelLoader
import torch
import gc
from logger_handler import logger

model_loader = ModelLoader()  # Create the model loader


class ModelUserClass:
    def __init__(self, llama_cpp_model_load):  # Inject the model loader
        self.llama_cpp_model_load = llama_cpp_model_load

    def use_model(self):
        # Get the model from the provided loader
        return self.llama_cpp_model_load.get_model()


model_pipeline = ModelUserClass(model_loader)
pipeline = model_pipeline.use_model()


st.title("Intics Chatbot")


if "messages" not in st.session_state:
    st.session_state.messages = []

for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])


def chat_prompt(prompt: str):
    try:
        prompt_val = prompt

        messages = [
            {"role": "system", "content": prompt_val},
            {"role": "user", "content": ""},
        ]

        # response = pipeline.create_chat_completion(messages=messages, response_format={"type": "json_object"})
        response = pipeline.create_chat_completion(messages=messages)

        prompt_result = response["choices"][0]["message"]["content"].strip()
        # prompt_result = response

        logger.info("completed response generation from llm")

        return prompt_result

    except Exception as ex:
        raise ex
    finally:
        gc.collect()
        torch.cuda.empty_cache()


if instruction := st.chat_input("Ask me..."):

    st.session_state.messages.append({"role": "user", "content": instruction})

    with st.chat_message("user"):
        st.markdown(instruction)

    # url = 'http://192.168.10.238:10002/chat/llama'
    # headers = {'accept': 'application/json'}
    # with st.chat_message("assistant"):
    #     response = requests.post(url, headers=headers, params={'prompt': instruction})
    #     prompt_result = response.
    #     st.write(prompt_result)

    prompt_result = chat_prompt(instruction)
    st.session_state.messages.append({"role": "assistant", "content": prompt_result})
