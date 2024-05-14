import streamlit as st
from model_load import ModelLoad
model, tokenizer = ModelLoad.krypton_chat_4bit_model_load()
from transformers import TextStreamer


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
        text_streamer = TextStreamer(tokenizer)
        outputs = model.generate(input_ids=prompt, streamer=text_streamer, max_new_tokens=2048, use_cache=True, pad_token_id=tokenizer.pad_token_id)
        prompt_result = tokenizer.batch_decode(outputs, skip_special_tokens=True)

        response = st.markdown(prompt_result)
    st.session_state.messages.append({"role": "assistant", "content": response})
