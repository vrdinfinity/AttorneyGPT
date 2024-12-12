import streamlit as st
import time
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain.prompts import PromptTemplate
from langchain_together import Together
from dotenv import load_dotenv
load_dotenv()
import os
from langchain.memory import ConversationBufferWindowMemory
from langchain.chains import ConversationalRetrievalChain

st.set_page_config(page_title="AttorneyGPT", layout="wide")



col1, col2, col3 = st.columns([1, 4, 1])
with col2:
    st.image("AttorneyGPTcover.png")

# Add welcome message at the top
with col2:
    st.markdown("<h2 style='text-align: center; color: #007bff;'>Welcome to AttorneyGPT!</h2>", unsafe_allow_html=True)
    st.markdown(
        """
        <div style='text-align: center; color: #e0e0e0;'>
            Attorney GPT is your specialized AI assistant for navigating the Indian Penal Code. Just type in your questions, and it will provide precise, professional responses based on the IPC.
        </div>
        """, 
        unsafe_allow_html=True
    )

st.markdown(
    """
    <style>
    /* General background color and text color */
    .reportview-container {
        background-color: #1e1e1e; /* Dark background */
        color: #e0e0e0; /* Light gray text */
        font-family: 'Arial', sans-serif; /* Match ChatGPT font */
    }

    /* Sidebar styling */
    .css-1g7bq04 {
        background-color: #2e2e2e; /* Dark gray sidebar */
    }

    /* Button styling */
    div.stButton > button:first-child {
        background-color: #007bff; /* Blue button */
        color: #ffffff; /* White text */
        border: none;
        border-radius: 4px;
        padding: 10px 20px;
    }
    div.stButton > button:active {
        background-color: #0056b3; /* Darker blue */
    }

    /* Input box styling */
    .css-1v0mbdj {
        background-color: #2c2c2c; /* Darker input background */
        color: #e0e0e0; /* Light gray text */
        border-radius: 4px;
        padding: 10px;
    }

    /* Message bubble styling */
    .chat-message {
        border-radius: 10px;
        padding: 10px;
        margin-bottom: 10px;
        max-width: 80%;
        line-height: 1.5;
    }
    .chat-message.user {
        background-color: #007bff; /* User messages in blue */
        color: #ffffff;
        align-self: flex-end;
    }
    .chat-message.assistant {
        background-color: #333333; /* Assistant messages in dark gray */
        color: #e0e0e0;
        align-self: flex-start;
    }

    /* Hide unnecessary elements */
    #MainMenu {visibility: hidden;}
    .stDeployButton {display: none;}
    footer {visibility: hidden;}
    #stDecoration {display: none;}
    button[title="View fullscreen"] {visibility: hidden;}

    /* Chat container styling */
    .css-18e3th9 {
        border-radius: 8px;
        padding: 20px;
        background-color: #2c2c2c; /* Darker chat background */
    }
    </style>
    """,
    unsafe_allow_html=True
)

def reset_conversation():
    st.session_state.messages = []
    st.session_state.memory.clear()

if "messages" not in st.session_state:
    st.session_state.messages = []

if "memory" not in st.session_state:
    st.session_state.memory = ConversationBufferWindowMemory(k=2, memory_key="chat_history",return_messages=True) 

embeddings = HuggingFaceEmbeddings(model_name="nomic-ai/nomic-embed-text-v1", model_kwargs={"trust_remote_code":True, "revision":"289f532e14dbbbd5a04753fa58739e9ba766f3c7"})
db = FAISS.load_local("ipc_vector_db", embeddings, allow_dangerous_deserialization=True)
db_retriever = db.as_retriever(search_type="similarity", search_kwargs={"k": 4})

prompt_template = """<s>[INST]This is a chat template and As a legal chat bot specializing in Indian Penal Code queries, your primary objective is to provide accurate and concise information based on the user's questions. Do not generate your own questions and answers. You will adhere strictly to the instructions provided, offering relevant context from the knowledge base while avoiding unnecessary details. Your responses will be brief, to the point, and in compliance with the established format. If a question falls outside the given context, you will refrain from utilizing the chat history and instead rely on your own knowledge base to generate an appropriate response. You will prioritize the user's query and refrain from posing additional questions. The aim is to deliver professional, precise, and contextually relevant information pertaining to the Indian Penal Code. If the user greets you just greet back dont give any law information realted to it. and provide a prompt and example when welcoming.Don't tell gender neutral pronouns everytime while welcoming explain this app in a layman way instead.
CONTEXT: {context}
CHAT HISTORY: {chat_history}
QUESTION: {question}
ANSWER:
</s>[INST]
"""

prompt = PromptTemplate(template=prompt_template, input_variables=['context', 'question', 'chat_history'])

# You can also use other LLMs options from https://python.langchain.com/docs/integrations/llms. Here I have used TogetherAI API
TOGETHER_AI_API = os.environ['TOGETHER_AI_API_KEY']
llm = Together(
    model="mistralai/Mistral-7B-Instruct-v0.2",
    temperature=0.5,
    max_tokens=1024,
    together_api_key=f"{TOGETHER_AI_API}"
)

qa = ConversationalRetrievalChain.from_llm(
    llm=llm,
    memory=st.session_state.memory,
    retriever=db_retriever,
    combine_docs_chain_kwargs={'prompt': prompt}
)

for message in st.session_state.messages:
    role = message.get("role")
    with st.chat_message(role):
        st.markdown(f'<div class="chat-message {role}">{message.get("content")}</div>', unsafe_allow_html=True)

input_prompt = st.chat_input("Say something")

if input_prompt:
    with st.chat_message("user"):
        st.markdown(f'<div class="chat-message user">{input_prompt}</div>', unsafe_allow_html=True)

    st.session_state.messages.append({"role":"user","content":input_prompt})

    with st.chat_message("assistant"):
        with st.status("Thinking 💡...", expanded=True):
            result = qa.invoke(input=input_prompt)

            message_placeholder = st.empty()

            full_response = "⚠️ **Note: Information provided may be inaccurate.** \n\n\n"
            for chunk in result["answer"]:
                full_response += chunk
                time.sleep(0.02)
                message_placeholder.markdown(f'<div class="chat-message assistant">{full_response} ▌</div>', unsafe_allow_html=True)
                
        st.button('Reset All Chat 🗑️', on_click=reset_conversation)

    st.session_state.messages.append({"role":"assistant","content":result["answer"]})