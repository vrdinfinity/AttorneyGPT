import streamlit as st
import time
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain.prompts import PromptTemplate
from langchain_together import Together
from dotenv import load_dotenv
from gtts import gTTS
import os
import base64
from langchain.memory import ConversationBufferWindowMemory
from langchain.chains import ConversationalRetrievalChain
import speech_recognition as sr
import sounddevice as sd
import scipy.io.wavfile as wav
import numpy as np
import tempfile
from deep_translator import GoogleTranslator

# Page configuration
st.set_page_config(
    page_title="AttorneyGPT",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# Load environment variables
load_dotenv()

# Custom CSS
st.markdown(
    """
    <style>
    .reportview-container {
        background-color: #1e1e1e;
        color: #e0e0e0;
        font-family: 'Arial', sans-serif;
    }

    .css-1g7bq04 {
        background-color: #2e2e2e;
    }

    div.stButton > button:first-child {
        background-color: #007bff;
        color: #ffffff;
        border: none;
        border-radius: 4px;
        padding: 10px 20px;
    }

    div.stButton > button:active {
        background-color: #0056b3;
    }

    .css-1v0mbdj {
        background-color: #2c2c2c;
        color: #e0e0e0;
        border-radius: 4px;
        padding: 10px;
    }

    .chat-message {
        border-radius: 10px;
        padding: 10px;
        margin-bottom: 10px;
        max-width: 80%;
        line-height: 1.5;
    }

    .chat-message.user {
        background-color: #007bff;
        color: #ffffff;
        align-self: flex-end;
    }

    .chat-message.assistant {
        background-color: #333333;
        color: #e0e0e0;
        align-self: flex-start;
    }

    .voice-input-container {
        display: flex;
        align-items: center;
        gap: 10px;
        margin: 10px 0;
    }

    .record-button {
        background-color: #dc3545;
        color: white;
        border: none;
        border-radius: 50%;
        width: 40px;
        height: 40px;
        display: flex;
        align-items: center;
        justify-content: center;
        cursor: pointer;
        transition: background-color 0.3s;
    }

    .language-button {
        background-color: #4CAF50;
        border: none;
        color: white;
        padding: 10px 20px;
        text-align: center;
        text-decoration: none;
        display: inline-block;
        font-size: 16px;
        margin: 4px 2px;
        cursor: pointer;
        border-radius: 4px;
    }

    .language-button.selected {
        background-color: #45a049;
        box-shadow: 0 0 5px #fff;
    }

    .language-container {
        display: flex;
        justify-content: center;
        gap: 10px;
        margin: 20px 0;
    }

    #MainMenu {visibility: hidden;}
    .stDeployButton {display: none;}
    footer {visibility: hidden;}
    #stDecoration {display: none;}
    button[title="View fullscreen"] {visibility: hidden;}
    </style>
    """,
    unsafe_allow_html=True
)

# Initialize session states
if "messages" not in st.session_state:
    st.session_state.messages = []

if "memory" not in st.session_state:
    st.session_state.memory = ConversationBufferWindowMemory(
        k=2,
        memory_key="chat_history",
        return_messages=True
    )

if 'audio_responses' not in st.session_state:
    st.session_state.audio_responses = []

if 'recording' not in st.session_state:
    st.session_state.recording = False

if 'audio_data' not in st.session_state:
    st.session_state.audio_data = None

if 'selected_language' not in st.session_state:
    st.session_state.selected_language = 'English'

# Language configurations
LANGUAGE_CONFIG = {
    'English': {'code': 'en', 'gtts_code': 'en', 'name': 'English'},
    'Tamil': {'code': 'ta', 'gtts_code': 'ta', 'name': 'Tamil'},
    'Hindi': {'code': 'hi', 'gtts_code': 'hi', 'name': 'Hindi'}
}

def record_audio(duration=5):
    """Record audio for a specified duration with improved handling"""
    try:
        sample_rate = 16000
        channels = 1
        dtype = np.int16
        
        status_message = st.empty()
        
        recording = sd.rec(
            int(duration * sample_rate),
            samplerate=sample_rate,
            channels=channels,
            dtype=dtype
        )
        
        for remaining in range(duration, 0, -1):
            status_message.info(f"Recording... {remaining} seconds remaining")
            time.sleep(1)
        status_message.empty()
        
        sd.wait()
        
        recording = np.squeeze(recording)
        max_int16 = 2**15
        recording = np.clip(recording * max_int16, -max_int16, max_int16-1).astype(np.int16)
        
        return recording, sample_rate
    except Exception as e:
        st.error(f"Error recording audio: {str(e)}")
        return None, None
    
def convert_audio_to_text(audio_data, sample_rate):
    """Convert audio data to text with improved error handling"""
    temp_audio_path = None
    try:
        if audio_data is None or sample_rate is None:
            raise ValueError("Invalid audio data or sample rate")
            
        # Create a temporary WAV file
        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as temp_audio:
            temp_audio_path = temp_audio.name
            
            # Ensure audio_data is the correct shape
            if len(audio_data.shape) == 2:
                audio_data = audio_data.flatten()
            
            # Write WAV file
            wav.write(temp_audio_path, sample_rate, audio_data)
            
            # Initialize speech recognition
            recognizer = sr.Recognizer()
            recognizer.energy_threshold = 300
            recognizer.dynamic_energy_threshold = True
            recognizer.pause_threshold = 0.8
            
            with sr.AudioFile(temp_audio_path) as source:
                # Adjust for ambient noise
                recognizer.adjust_for_ambient_noise(source, duration=0.5)
                audio = recognizer.record(source)
                
                try:
                    # Try multiple speech recognition services
                    try:
                        text = recognizer.recognize_google(audio)
                    except sr.RequestError:
                        try:
                            text = recognizer.recognize_sphinx(audio)
                        except sr.RequestError:
                            raise sr.RequestError(
                                "All speech recognition services failed"
                            )
                    
                    return text
                except sr.UnknownValueError:
                    st.warning("Could not understand the audio. Please speak clearly and try again.")
                    return None
                except sr.RequestError as e:
                    st.error(f"Speech recognition service error: {str(e)}")
                    return None
                
    except Exception as e:
        st.error(f"Error converting speech to text: {str(e)}")
        return None
    finally:
        # Clean up temporary file
        if temp_audio_path and os.path.exists(temp_audio_path):
            try:
                os.remove(temp_audio_path)
            except:
                pass

def translate_text(text, source_lang='en', target_lang='en'):
    """Translate text between languages using deep-translator"""
    try:
        if source_lang == target_lang:
            return text
            
        max_chunk_length = 4500
        text_chunks = [text[i:i + max_chunk_length] for i in range(0, len(text), max_chunk_length)]
        
        translated_chunks = []
        for chunk in text_chunks:
            translator = GoogleTranslator(source=source_lang, target=target_lang)
            translated_chunk = translator.translate(chunk)
            translated_chunks.append(translated_chunk)
            
        return ' '.join(translated_chunks)
    except Exception as e:
        st.error(f"Translation error: {str(e)}")
        return text

def generate_audio(text, index, language='English'):
    """Generate audio file from text in specified language"""
    audio_path = None
    try:
        clean_text = text.replace('*', '').replace('_', '').replace('#', '')
        
        tts = gTTS(text=clean_text, lang=LANGUAGE_CONFIG[language]['gtts_code'])
        audio_path = f'response_audio_{index}.mp3'
        tts.save(audio_path)
        
        with open(audio_path, 'rb') as audio_file:
            audio_bytes = audio_file.read()
        audio_base64 = base64.b64encode(audio_bytes).decode()
        
        audio_html = f'''
            <audio controls>
                <source src="data:audio/mp3;base64,{audio_base64}" type="audio/mp3">
                Your browser does not support the audio element.
            </audio>
        '''
        return audio_html
    except Exception as e:
        st.error(f"Error generating audio: {str(e)}")
        return None
    finally:
        if audio_path and os.path.exists(audio_path):
            try:
                os.remove(audio_path)
            except:
                pass

def reset_conversation():
    """Reset all conversation state"""
    st.session_state.messages = []
    st.session_state.memory.clear()
    st.session_state.audio_responses = []

# Initialize embeddings and database
try:
    embeddings = HuggingFaceEmbeddings(
        model_name="nomic-ai/nomic-embed-text-v1",
        model_kwargs={
            "trust_remote_code": True,
            "revision": "289f532e14dbbbd5a04753fa58739e9ba766f3c7"
        }
    )
    db = FAISS.load_local("ipc_vector_db", embeddings, allow_dangerous_deserialization=True)
    db_retriever = db.as_retriever(search_type="similarity", search_kwargs={"k": 4})
except Exception as e:
    st.error(f"Error loading embeddings or database: {str(e)}")
    st.stop()

# Initialize LLM and prompt
prompt_template = """<s>[INST]This is a chat template and As a legal chat bot specializing in Indian Penal Code queries, your primary objective is to provide accurate and concise information based on the user's questions. Do not generate your own questions and answers. You will adhere strictly to the instructions provided, offering relevant context from the knowledge base while avoiding unnecessary details. Your responses will be brief, to the point, and in compliance with the established format. If a question falls outside the given context, you will refrain from utilizing the chat history and instead rely on your own knowledge base to generate an appropriate response. You will prioritize the user's query and refrain from posing additional questions. The aim is to deliver professional, precise, and contextually relevant information pertaining to the Indian Penal Code. If the user greets you greet them back and dont generate anything extra but tell them how can you be useful in a short manner and dont tell anything about any case when someone greets you.
CONTEXT: {context}
CHAT HISTORY: {chat_history}
QUESTION: {question}
ANSWER:
</s>[INST]
"""

prompt = PromptTemplate(
    template=prompt_template,
    input_variables=['context', 'question', 'chat_history']
)

# Initialize LLM
try:
    TOGETHER_AI_API = os.environ['TOGETHER_AI_API_KEY']
    llm = Together(
        model="mistralai/Mistral-7B-Instruct-v0.2",
        temperature=0.5,
        max_tokens=1024,
        together_api_key=TOGETHER_AI_API
    )
except Exception as e:
    st.error(f"Error initializing LLM: {str(e)}")
    st.stop()

# Initialize QA chain
qa = ConversationalRetrievalChain.from_llm(
    llm=llm,
    memory=st.session_state.memory,
    retriever=db_retriever,
    combine_docs_chain_kwargs={'prompt': prompt}
)

# Header and logo
col1, col2, col3 = st.columns([1, 4, 1])
with col2:
    st.image("AttorneyGPTcover.png")

# Add language selection buttons
st.markdown("<div class='language-container'>", unsafe_allow_html=True)
col1, col2, col3 = st.columns([1, 1, 1])
with col1:
    if st.button('English', key='en_button', 
                 help="Switch to English",
                 type="primary" if st.session_state.selected_language == 'English' else "secondary"):
        st.session_state.selected_language = 'English'
with col2:
    if st.button('தமிழ்', key='ta_button', 
                 help="Switch to Tamil",
                 type="primary" if st.session_state.selected_language == 'Tamil' else "secondary"):
        st.session_state.selected_language = 'Tamil'
with col3:
    if st.button('हिंदी', key='hi_button', 
                 help="Switch to Hindi",
                 type="primary" if st.session_state.selected_language == 'Hindi' else "secondary"):
        st.session_state.selected_language = 'Hindi'
st.markdown("</div>", unsafe_allow_html=True)

# Display existing messages
for message in st.session_state.messages:
    role = message.get("role")
    with st.chat_message(role):
        st.markdown(
            f'<div class="chat-message {role}">{message.get("content")}</div>',
            unsafe_allow_html=True
        )
        if role == "assistant":
            index = len([m for m in st.session_state.messages[:st.session_state.messages.index(message)] if m["role"] == "assistant"])
            if index < len(st.session_state.audio_responses):
                st.markdown(st.session_state.audio_responses[index], unsafe_allow_html=True)

# Create input container
input_container = st.container()

# Create two columns for text and voice input
with input_container:
    try:
        col1, col2 = st.columns([4, 1])
        
        with col1:
            input_prompt = st.chat_input("Ask your legal question...", key="chat_input")
        
        with col2:
            if st.button("🎤 Record", key="record_button", help="Click to record your question (5 seconds)"):
                audio_data, sample_rate = record_audio()
                if audio_data is not None:
                    transcribed_text = convert_audio_to_text(audio_data, sample_rate)
                    if transcribed_text:
                        st.info(f"Transcribed: {transcribed_text}")
                        input_prompt = transcribed_text
    except Exception as e:
        st.error(f"Error in input handling: {str(e)}")

if input_prompt:
    try:
        # Display user message
        with st.chat_message("user"):
            # Detect source language and translate to English if needed
            if st.session_state.selected_language != 'English':
                source_lang = LANGUAGE_CONFIG[st.session_state.selected_language]['code']
                english_input = translate_text(input_prompt, source_lang, 'en')
            else:
                english_input = input_prompt
                
            st.markdown(
                f'<div class="chat-message user">{input_prompt}</div>',
                unsafe_allow_html=True
            )
        st.session_state.messages.append({"role": "user", "content": input_prompt})

        # Generate and display assistant response
        with st.chat_message("assistant"):
            try:
                with st.status("Thinking 💡...", expanded=True):
                    result = qa.invoke(input=english_input)
                    message_placeholder = st.empty()
                    
                    # Translate response if needed
                    if st.session_state.selected_language != 'English':
                        target_lang = LANGUAGE_CONFIG[st.session_state.selected_language]['code']
                        translated_response = translate_text(
                            result["answer"],
                            'en',
                            target_lang
                        )
                    else:
                        translated_response = result["answer"]
                    
                    full_response = "⚠️ **_Note: Information provided may be inaccurate._** \n\n\n"
                    full_response += translated_response
                    message_placeholder.markdown(
                        f'<div class="chat-message assistant">{full_response}</div>',
                        unsafe_allow_html=True
                    )
                
                # Generate and display audio in selected language
                audio_html = generate_audio(
                    translated_response, 
                    len(st.session_state.audio_responses),
                    st.session_state.selected_language
                )
                if audio_html:
                    st.session_state.audio_responses.append(audio_html)
                    st.markdown(audio_html, unsafe_allow_html=True)
                
                # Add assistant message to history
                st.session_state.messages.append({
                    "role": "assistant",
                    "content": translated_response
                })
                
            except Exception as e:
                st.error(f"Error generating response: {str(e)}")
            
            # Reset button
            st.button('Reset All Chat 🗑️', on_click=reset_conversation)
            
    except Exception as e:
        st.error(f"Error processing input: {str(e)}")