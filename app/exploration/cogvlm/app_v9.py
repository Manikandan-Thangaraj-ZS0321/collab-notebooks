import streamlit as st
import json
import os
import pandas as pd
import ast
import time

# audio recorder package
from audiorecorder import audiorecorder

# video recorder package
from streamlit_webrtc import webrtc_streamer, WebRtcMode
from src.video.video_stream import AudioRecorderProcessor

# upload file saver
from src.uploder.upload_file_saver import save_uploaded_file

# speech to text process
from src.voice.speech2text import perform_automatic_speech_recognition

# llm model
from src.llm.mistral_model import llm_model

# post-process
from src.process.postprocess import response_postprocess, extract_curl_command

# trello api
from src.trelloAPI.curl_processor import execute_curl
from src.trelloAPI.desc_reader import read_file_contents
from src.trelloAPI.suggestion_provider import suggestion_providers

# image extraction

from src.image.image_to_text import generate_tokens_paddle
from src.image.mindee_text import mindee_ocr

# pdf extraction

from src.pdf.pdf_splitter import split_pdf
from src.pdf.pdf_to_text import pdf_text_extraction
from src.pdf.pdf_to_image import pdf_to_images

# table attribution xenon

from table_attribution import table_attribution_yolo

# text to speech process
# from src.voice.text2speech import perform_text_2_speech

st.set_page_config(page_title="intics", page_icon="✨")

with st.sidebar:
    st.title("InticsGPT 1.0")

    st.markdown("---")
    st.subheader(" 🛠️ Sevices")
    option = st.sidebar.selectbox(
        'Choose service',
        ('prompt', 'Trello', 'Table Extraction', 'KVP Extraction')
    )

    trello_api_active = None
    trello_curl_executor = None
    table_xenon_enable = None

    if option == 'Trello':
        trello_api_active = True

    elif option == 'Table Extraction':
        table_xenon_enable = st.checkbox("Xenon", key='table-xenon')

    st.markdown("---")
    st.subheader(" 💻 Upload File")
    uploaded_files = st.file_uploader("Choose files", accept_multiple_files=True)

    if uploaded_files:
        file_type = uploaded_files[0].type
        if file_type == 'application/pdf':
            selected_page = st.number_input("pdf page", value=1, step=1, format="%d", key="number_input")

    st.markdown("---")
    st.subheader("🎙️ Audio Recorder")
    audio = []
    audio_enable = st.checkbox("audio_enable")
    if audio_enable:
        audio = audiorecorder("🔊 Start Recorder", "🔈 Stop Recorder")
        # voice_prompt = st.text_input("Enter your prompt:", value='', key='voice')
        voice_prompt = ''
    st.markdown("---")
    st.subheader(" 📹 Video Recorder")
    recorder_state = webrtc_streamer(
        key="audio-recorder",
        mode=WebRtcMode.SENDRECV,
        audio_processor_factory=AudioRecorderProcessor,
    )
    st.markdown("---")

st.title("How can I help you today?")

trello_instruction = '''you are a coder. please only write curl command for trello api. API KEY IS 
0be7cc52baffadc73fea8ef79e5ad293, ACCESS TOKEN is 
ATTAf7baa208a7c1c50e9a15061b2603a707076ed47992306f021e73a99251271cefDD71D635'''

desc_instruction = ''' Read what is given in paragraph 2. Go and search in paragraph 1 and if you can find anything related to what you read there get me the result. Don't add extra information. Get the same text that you see in paragraph 1. Give your answer in bullet points brief . i need top 10 items only '''

# chat conversation with text

if "messages" not in st.session_state:
    st.session_state.messages = []

if "button" not in st.session_state:
    st.session_state.button = []

if "id" not in st.session_state:
    st.session_state.id = []

for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

instruction = st.chat_input("Your message")

if len(uploaded_files) == 0:

    if prompt := instruction:

        with st.chat_message("user"):
            st.markdown(prompt)

        st.session_state.messages.append({"role": "user", "content": prompt})

        if trello_api_active:

            with st.spinner('Intics-Trello...'):

                desc_reader = read_file_contents("src/trelloAPI/trello_desc/trello_desc.txt")

                prompt_desc = f"PARAGRAPH 2 :\n" + str(prompt)

                desc_response = llm_model("", desc_reader + str(prompt_desc) + desc_instruction,
                                          max_new_tokens=1000)
                desc_response_process = response_postprocess(desc_response)

                with st.chat_message("assistant"):
                    st.markdown(desc_response_process)

                st.session_state.button.append({"role": "assistant", "content": desc_response_process})

        else:
            with st.spinner('Intics...'):

                llm_response = llm_model("", str(prompt), max_new_tokens=1000)
                response_process = response_postprocess(llm_response)

                with st.chat_message("assistant"):
                    st.markdown(response_process)

                st.session_state.messages.append({"role": "assistant", "content": response_process})

# upload files

if uploaded_files and instruction:

    with st.chat_message("user"):
        st.markdown(instruction)

    st.session_state.messages.append({"role": "user", "content": instruction})

    for uploaded_file in uploaded_files:

        file_type = uploaded_file.type

        if file_type == 'audio/wav' or file_type == 'audio/mpeg':

            save_dir = "data/audio_uploader"
            saved_file_path = save_uploaded_file(uploaded_file, save_dir)
            uploaded_file = None

            with st.spinner('Audio Processing...'):
                audio_transcribe = perform_automatic_speech_recognition(saved_file_path)

                with st.chat_message("user"):
                    st.markdown(audio_transcribe)

                st.session_state.messages.append({"role": "user", "content": audio_transcribe})

                if trello_api_active:

                    desc_reader = read_file_contents("src/trelloAPI/trello_desc/trello_desc.txt")

                    prompt_desc = f"PARAGRAPH 2 :\n" + str(audio_transcribe)

                    desc_response = llm_model(instruction, desc_reader + str(prompt_desc) + desc_instruction,
                                              max_new_tokens=1000)
                    desc_response_process = response_postprocess(desc_response)

                    with st.chat_message("assistant"):
                        st.markdown(desc_response_process)

                    st.session_state.button.append({"role": "assistant", "content": desc_response_process})

                else:

                    llm_response = llm_model(instruction, str(audio_transcribe), max_new_tokens=1000)
                    response_process = response_postprocess(llm_response)

                    with st.chat_message("assistant"):
                        st.markdown(response_process)

                    st.session_state.messages.append({"role": "assistant", "content": response_process})


        elif file_type == 'application/pdf':
            with st.spinner('Activating Document ...'):

                # pdf uploader
                save_dir = "data/pdf_uploader"
                saved_file_path = save_uploaded_file(uploaded_file, save_dir)

                # pdf splitter
                filename = os.path.splitext(os.path.basename(saved_file_path))[0]
                pdf_pages_save_dir = f"data/pdf_splitter/{filename}"
                if not os.path.exists(pdf_pages_save_dir):
                    os.makedirs(pdf_pages_save_dir)
                split_pdf(saved_file_path, pdf_pages_save_dir)

                for pdf_page in os.listdir(pdf_pages_save_dir):

                    pdf_file = os.path.join(pdf_pages_save_dir, pdf_page)
                    page_filename = os.path.splitext(os.path.basename(pdf_file))[0]

                    if str(page_filename) == str(selected_page):
                        extracted_text = pdf_text_extraction(pdf_file)
                        print(len(extracted_text))

                        if len(extracted_text)>10:

                            llm_response = llm_model(instruction, str(extracted_text), max_new_tokens=1000)
                            response_process = response_postprocess(llm_response)

                            with st.chat_message("assistant"):
                                st.markdown(response_process)

                            ax = '''if "{" in response_process and "}" in response_process:
                                res_json = json.loads(f'{response_process}')
                                if isinstance(res_json, dict):
                                    json_response = json.dumps(res_json)
                                    with st.chat_message("assistant"):
                                        st.json(json_response)'''

                            st.session_state.messages.append({"role": "assistant", "content": response_process})
                        else:
                            pdf_image_path = f"data/image_uploader/{filename}"

                            pdf_to_images(pdf_file, pdf_image_path)

                            for image_file in os.listdir(pdf_image_path):
                                if image_file.endswith(".jpg") or image_file.endswith(".png") or image_file.endswith(".jpeg"):

                                    extracted_text = generate_tokens_paddle(pdf_image_path+"/"+image_file)

                                    llm_response = llm_model("", str(extracted_text) + "\n \n" + instruction,
                                                             max_new_tokens=1000)
                                    response_process = response_postprocess(llm_response)

                                    with st.chat_message("assistant"):
                                        st.markdown(response_process)

                                    st.session_state.messages.append({"role": "assistant", "content": response_process})






        elif file_type == 'image/png' or 'image/jpeg':

            if not table_xenon_enable:
                with st.spinner('Activating Document ...'):
                    save_dir = "data/image_uploader"
                    saved_file_path = save_uploaded_file(uploaded_file, save_dir)
                    uploaded_file = None


                    def stream_data(extracted_text):
                        for word in extracted_text.split(" "):
                            yield word + " "
                            time.sleep(0.02)


                    extracted_text = generate_tokens_paddle(saved_file_path)
                    # extracted_text = mindee_ocr(saved_file_path)
                    # st.write_stream(stream_data(extracted_text))

                    llm_response = llm_model("", str(extracted_text) + "\n \n" + instruction, max_new_tokens=1000)
                    # llm_response = llm_model("Text Allignment", str(extracted_text) + "\n \n" + instruction, max_new_tokens=1000)
                    response_process = response_postprocess(llm_response)

                    with st.chat_message("assistant"):
                        st.markdown(response_process)

                    st.session_state.messages.append({"role": "assistant", "content": response_process})

            if table_xenon_enable:
                with st.spinner('Image AI...'):
                    save_dir = "data/table_uploader"
                    saved_file_path = save_uploaded_file(uploaded_file, save_dir)
                    uploaded_file = None

                    table_output_dir = 'data/tmpp/'
                    table_excel_path = table_attribution_yolo(saved_file_path, table_output_dir)
                    df = pd.read_excel(table_excel_path["table_csv"])
                    st.write(df)

# audio recorder

if len(audio) > 0 and audio is not None:
    st.audio(audio.export().read())

    saved_file_path = "data/audio_input.wav"
    audio.export(saved_file_path, format="wav")

    with st.spinner('Voice Processing...'):
        audio_transcribe = perform_automatic_speech_recognition(saved_file_path)

        with st.chat_message("user"):
            st.markdown(audio_transcribe)

        st.session_state.messages.append({"role": "user", "content": audio_transcribe})

        trello_instruction = '''please only write curl command using TRELLO CURL COMMANDS.
                        API KEY IS
                        0be7cc52baffadc73fea8ef79e5ad293, 
                        ACCESS TOKEN is
                        ATTAf7baa208a7c1c50e9a15061b2603a707076ed47992306f021e73a99251271cefDD71D635'''

        curl_reader = read_file_contents("src/trelloAPI/trello_desc/curl_desc.txt")

        id_no = ''

        if len(st.session_state.id) > 0:
            id_no = st.session_state.id[-1]['content']
            # id_no = st.session_state.id

        trello_llm_response = llm_model(trello_instruction,
                                        f'{curl_reader} {trello_instruction} {audio_transcribe} {id_no}',
                                        max_new_tokens=1000)

        trello_curl_command = response_postprocess(trello_llm_response)

        with st.chat_message("assistant"):
            st.markdown(trello_curl_command)

        st.session_state.messages.append({"role": "assistant", "content": trello_curl_command})

        output, error = execute_curl(str(trello_curl_command))

        with st.chat_message("assistant"):
            st.markdown(output)

        if output:

            output_instruction = "get the id without any extra content"
            # output_instruction = '''You need to extract the board ID and the ID of the list titled "Todo" from these JSON responses.'''
            id_data = llm_model(output_instruction,
                                f'{output}',
                                max_new_tokens=1000)
            id_data_postprocess = response_postprocess(id_data)

            try:
                # Parse JSON
                data = json.loads(id_data)

                # Find the "To Do" list and extract its ID
                todo_id = None
                for item in data:
                    if item['name'] == 'To Do':
                        todo_id = item['id']
                        break

                # id_data_postprocess = str(id_data_postprocess)
                # id_data_postprocess = id_data_postprocess.replace("'", '"')
                # list_a = ast.literal_eval(str(id_data_postprocess))[0]
                # print(list_a)
                # # st.session_state.id.append({"role": "assistant", "content": f'idlist:{list_a}'})
                st.session_state.id.append({"role": "assistant", "content": todo_id})
            except:
                list_a = id_data_postprocess
                print("loss")
                st.session_state.id.append({"role": "assistant", "content": list_a})

            with st.chat_message("assistant"):
                st.markdown(list_a)

            # suggestion = suggestion_providers(st.session_state.id[-1]['content'])
            st.session_state.messages.append({"role": "assistant", "content": output})

