import streamlit as st
import requests

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

    url = 'http://192.168.10.238:10002/chat/llama'
    headers = {'accept': 'application/json'}

    with st.spinner('Processing...'):
        with st.chat_message("assistant"):
            response = requests.post(url, headers=headers, params={'prompt': instruction})
            prompt_result = response.content
            st.write(response.content)

        st.session_state.messages.append({"role": "assistant", "content": prompt_result})
