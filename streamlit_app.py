import os
import streamlit as st
import openai
from PyPDF2 import PdfReader
from langchain.text_splitter import CharacterTextSplitter
from langchain_openai import OpenAIEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_openai import ChatOpenAI
from langchain.memory import ConversationBufferMemory
from langchain.chains import ConversationalRetrievalChain
from langdetect import detect
import logging

# Setup logging for unanswered questions
logging.basicConfig(filename='unanswered_questions.log', level=logging.INFO, format='%(asctime)s - %(message)s')

# Custom CSS for chat messages
css = '''
<style>
.chat-message {
    padding: 1rem;
    border-radius: 0.5rem;
    margin-bottom: 1rem;
    display: flex;
    justify-content: flex-start;
    align-items: flex-start;
}
.chat-message.user {
    background-color: #2b313e;
}
.chat-message.bot {
    background-color: #475063;
}
.chat-message .avatar {
    width: 20%;
    padding-right: 1rem;
}
.chat-message .avatar img {
    max-width: 100%;
    border-radius: 50%;
    object-fit: cover;
}
.chat-message .message {
    width: 80%;
    padding: 0.5rem;
    color: white;
    word-wrap: break-word;
}
</style>
'''

# HTML templates for chat messages
bot_template = '''
<div class="chat-message bot">
    <div class="avatar">
        <img src="https://i.ibb.co/sPvCDGT/Robot.jpg" alt="Robot">
    </div>
    <div class="message">{{MSG}}</div>
</div>
'''

user_template = '''
<div class="chat-message user">
    <div class="avatar">
        <img src="https://i.ibb.co/pPPVnfL/Human.jpg" alt="Human">
    </div>
    <div class="message">{{MSG}}</div>
</div>
'''

# Load the API key from Streamlit secrets
openai.api_key = st.secrets['OPENAI_API_KEY']
os.environ['OPENAI_API_KEY'] = st.secrets['OPENAI_API_KEY']

def get_pdf_text_from_folder(folder_path):
    text = ""
    for filename in os.listdir(folder_path):
        if filename.endswith(".pdf"):
            pdf_path = os.path.join(folder_path, filename)
            pdf_reader = PdfReader(pdf_path)
            for page in pdf_reader.pages:
                text += page.extract_text()
    return text

def get_text_chunks(text):
    text_splitter = CharacterTextSplitter(
        separator="\n",
        chunk_size=1000,
        chunk_overlap=200,
        length_function=len
    )
    chunks = text_splitter.split_text(text)
    return chunks

def get_vectorstore(text_chunks):
    embeddings = OpenAIEmbeddings(openai_api_key=openai.api_key)
    vectorstore = FAISS.from_texts(texts=text_chunks, embedding=embeddings)
    return vectorstore

def get_conversation_chain(vectorstore):
    llm = ChatOpenAI(openai_api_key=openai.api_key)
    memory = ConversationBufferMemory(
        memory_key='chat_history', return_messages=True)
    conversation_chain = ConversationalRetrievalChain.from_llm(
        llm=llm,
        retriever=vectorstore.as_retriever(),
        memory=memory
    )
    return conversation_chain

def handle_userinput(user_question):
    lang = detect(user_question)

    if lang == 'de':
        response = st.session_state.conversation({'question': user_question})
        answer = response['chat_history'][-1].content
        
        if "Ich weiß es nicht" in answer or "Es tut mir leid" in answer or "ich kann die Frage nicht beantworten" in answer:
            logging.info(f"Unanswered question: {user_question}")
        
        st.session_state.chat_history.append((user_question, answer))
    else:
        st.write(bot_template.replace("{{MSG}}", "Bitte stellen Sie Ihre Frage auf Deutsch."), unsafe_allow_html=True)

def admin_page():
    st.title("Admin Page")
    username = st.text_input("Username")
    password = st.text_input("Password", type="password")
    login_button = st.button("Login")

    if login_button:
        if username == "admin" and password == "admin123":
            st.success("Logged in successfully!")
            st.session_state["admin_logged_in"] = True
        else:
            st.error("Invalid username or password")

    if st.session_state.get("admin_logged_in"):
        st.subheader("Unanswered Questions Log")
        log_content = ""
        with open('unanswered_questions.log', 'r') as log_file:
            log_content = log_file.read()
        
        st.text_area("Log Content", log_content, height=300)
        st.download_button("Download Log", log_content, file_name='unanswered_questions.log')
        if st.sidebar.button("Back to Main Page"):
            st.session_state.page = "main"

def main_page():
    st.title("BIOGASANLAGE RAG CHATBOT")

    with st.spinner("Verarbeitung von PDFs..."):
        raw_text = get_pdf_text_from_folder('data')
        text_chunks = get_text_chunks(raw_text)
        vectorstore = get_vectorstore(text_chunks)
        st.session_state.conversation = get_conversation_chain(vectorstore)

    user_question = st.chat_input("Stellen Sie eine Frage:")
    if user_question:
        handle_userinput(user_question)

    if st.session_state.chat_history:
        for question, answer in st.session_state.chat_history:
            st.write(user_template.replace("{{MSG}}", question), unsafe_allow_html=True)
            st.write(bot_template.replace("{{MSG}}", answer), unsafe_allow_html=True)

    if st.sidebar.button("Admin Page"):
        st.session_state.page = "admin"

def main():
    st.set_page_config(page_title="FULDA BIOGASANLAGE RAG CHATBOT", page_icon=":robot:")
    st.write(css, unsafe_allow_html=True)

    if "conversation" not in st.session_state:
        st.session_state.conversation = None
    if "chat_history" not in st.session_state:
        st.session_state.chat_history = []
    if "page" not in st.session_state:
        st.session_state.page = "main"

    if st.session_state.page == "main":
        main_page()
    elif st.session_state.page == "admin":
        admin_page()

if __name__ == '__main__':
    main()

