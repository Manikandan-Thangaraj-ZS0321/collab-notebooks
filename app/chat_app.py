import streamlit as st
from model_load import ModelLoad
# pipeline = ModelLoad.krypton_chat_llamacpp_model_load()


st.title("LLama3")


if "messages" not in st.session_state:
    st.session_state.messages = []

for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

if prompt := st.chat_input("What is up?"):
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    messages = [
        {"role": "system", "content": ""},
        {"role": "user", "content": prompt},
    ]

    with st.chat_message("assistant"):
        # stream = client.chat.completions.create(
        #     model=st.session_state["openai_model"],
        #     messages=[
        #         {"role": m["role"], "content": m["content"]}
        #         for m in st.session_state.messages
        #     ],
        #     stream=True,
        # )
        response_val = ModelLoad.krypton_chat_llamacpp_model_load().create_chat_completion(messages=messages, stream=True)
        prompt_result = response_val["choices"][0]["message"]["content"].strip()

        response = st.write_stream(response_val)
    st.session_state.messages.append({"role": "assistant", "content": response})
