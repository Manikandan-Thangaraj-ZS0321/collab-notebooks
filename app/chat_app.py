import streamlit as st
from llama3_llama_cpp import chat_prompt

st.title("Intics Chatbot")


if "messages" not in st.session_state:
    st.session_state.messages = []

for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])


if instruction := st.chat_input("Ask me..."):

    st.session_state.messages.append({"role": "user", "content": instruction})

    with st.chat_message("user"):
        st.markdown(instruction)

    with st.chat_message("assistant"):
        prompt_result = chat_prompt(instruction)
        st.write(prompt_result)
    st.session_state.messages.append({"role": "assistant", "content": prompt_result})
