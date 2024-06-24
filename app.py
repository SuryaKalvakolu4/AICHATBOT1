import os
import streamlit as st
import openai
from dotenv import load_dotenv
from PyPDF2 import PdfReader
from langchain.text_splitter import CharacterTextSplitter
from langchain_openai import OpenAIEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_openai import ChatOpenAI
from langchain.memory import ConversationBufferMemory
from langchain.chains import ConversationalRetrievalChain
from htmlTemplates import css, bot_template, user_template
import logging
from langdetect import detect

# Load the API key from Streamlit secrets
openai.api_key = st.secrets['OPENAI_API_KEY']
os.environ['OPENAI_API_KEY'] = st.secrets['OPENAI_API_KEY']

# Set up logging
logging.basicConfig(filename='unanswered_questions.log', level=logging.INFO,
                    format='%(asctime)s - %(message)s')  # Customize format if needed

logger = logging.getLogger()

# Custom filter to identify unanswered questions
class UnansweredQuestionFilter(logging.Filter):
    def filter(self, record):
        return "Unanswered question" in record.getMessage()

logger.addFilter(UnansweredQuestionFilter())

def log_unanswered_question(question):
    logger.info(f"Unanswered question: {question}")

def get_pdf_text_from_folder(folder_path):
    text = ""
    for filename in os.listdir(folder_path):
        if filename.endswith(".pdf"):
            pdf_path = os.path.join(folder_path, filename)
            pdf_reader = PdfReader(os.path.join(folder_path, filename))
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
    # Detect the language of the user's question
    lang = detect(user_question)

    if lang == 'de':
        response = st.session_state.conversation({'question': user_question})
        st.session_state.chat_history = response['chat_history']

        answered = False
        for i, message in enumerate(st.session_state.chat_history):
            if i % 2 == 0:
                st.write(user_template.replace(
                    "{{MSG}}", message.content), unsafe_allow_html=True)
            else:
                st.write(bot_template.replace(
                    "{{MSG}}", message.content), unsafe_allow_html=True)
                if any(response in message.content for response in ["Es tut mir leid", "Ich weiß es nicht", "Entschuldigung"]):
                    answered = False
                else:
                    answered = True
        
        if not answered:
            log_unanswered_question(user_question)
    else:
        st.write(bot_template.replace(
            "{{MSG}}", "Bitte stellen Sie Ihre Frage auf Deutsch."), unsafe_allow_html=True)

def admin_page():
    st.title("Verwaltungsseite")
    username = st.text_input("Benutzername")
    password = st.text_input("Passwort", type="password")
    login_button = st.button("Login")

    if login_button:
        if username == "admin" and password == "admin123":
            st.success("Logged in successfully!")
            st.session_state["admin_logged_in"] = True
        else:
            st.error("Ungültiger Benutzername oder Passwort")

    if st.session_state.get("admin_logged_in"):
        st.subheader("Logbuch der unbeantworteten Fragen")
        log_content = ""
        with open('unanswered_questions.log', 'r') as log_file:
            log_content = log_file.read()
        
        st.text_area("Log Content", log_content, height=300)

        st.download_button("Download Log", log_content, file_name='unanswered_questions.log')

def main():
    st.set_page_config(page_title="Chatten mit AI Chatbot", page_icon=":robot:")
    st.write(css, unsafe_allow_html=True)

    if "conversation" not in st.session_state:
        st.session_state.conversation = None
    if "chat_history" not in st.session_state:
        st.session_state.chat_history = []
    if "admin_logged_in" not in st.session_state:
        st.session_state.admin_logged_in = False

    st.sidebar.subheader("Ihre Dokumente")
    st.sidebar.button("Verwaltungsseite", on_click=lambda: st.session_state.update(show_admin_page=True))

    # Check for admin page display
    if st.session_state.get("show_admin_page"):
        admin_page()
        return

    st.title("FULDA BIOGASANLAGE CHATBOT")

    # Load PDFs from the data folder
    with st.spinner("Processing"):
        raw_text = get_pdf_text_from_folder('data')

        # Get the text chunks
        text_chunks = get_text_chunks(raw_text)

        # Create vector store
        vectorstore = get_vectorstore(text_chunks)

        # Create conversation chain
        st.session_state.conversation = get_conversation_chain(vectorstore)

    with st.spinner("Laden..."):
        user_question = st.text_input("Stellen Sie eine Frage zu Ihren Dokumenten:")
        if st.button("Laufen lassen"):
            if user_question:
                st.session_state.chat_history.append(user_question)
                handle_userinput(user_question)

    # Display previously asked questions
    if st.session_state.chat_history:
        st.sidebar.subheader("Zuvor gestellte Fragen")
        for question in st.session_state.chat_history:
            st.sidebar.write(question)

if __name__ == '__main__':
    main()
