import streamlit as st
from model_load import ModelLoad
from llama3_llama_cpp import chat_prompt

st.title("LLama3")


if "messages" not in st.session_state:
    st.session_state.messages = []

for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# if prompt := st.chat_input("What is up?"):
#     st.session_state.messages.append({"role": "user", "content": prompt})
#     with st.chat_message("user"):
#         st.markdown(prompt)


if instruction := st.chat_input("Your message"):
    messages = [
        {"role": "system", "content": ""},
        {"role": "user", "content": instruction},
    ]
    with st.chat_message("user"):
        st.markdown(instruction)

    st.session_state.messages.append({"role": "user", "content": instruction})

    with st.chat_message("assistant"):
        # stream = client.chat.completions.create(
        #     model=st.session_state["openai_model"],
        #     messages=[
        #         {"role": m["role"], "content": m["content"]}
        #         for m in st.session_state.messages
        #     ],
        #     stream=True,
        # )
        # response_val = ModelLoad.krypton_chat_llamacpp_model_load().create_chat_completion(messages=messages, stream=True)
        prompt_result = chat_prompt(instruction)

    st.session_state.messages.append({"role": "assistant", "content": prompt_result})
