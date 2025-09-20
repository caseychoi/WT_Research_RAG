import os
import streamlit as st
import tempfile
import uuid
import shutil
import re
import fitz  # PyMuPDF library

from langchain_core.messages import AIMessage, HumanMessage
from langchain.chains.retrieval import create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_community.document_loaders import PyPDFLoader
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.chat_models import ChatOllama
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.documents import Document
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# --- Configuration and Constants ---
APP_TITLE = "WT Research RAG"
FAISS_INDEX_PATH = "faiss_index"

LLM_SYSTEM_PROMPT_BASE = """
You are a spiritually respectful, detailed, and formal research assistant. Your purpose is to provide scripturally-based answers using only official Jehovah's Witnesses publications provided in the context below. You must not use any external knowledge.

General Guidelines:
- **Source Material:** First, check the updated knowledge found in the uploaded PDF documents.
- **Citations and Links:** Use direct quotations from source articles or publications. Always include a hot link to the original article, study note, or scripture. If a page or paragraph number is available (e.g., w21.01 p.22, par.10), include it clearly.
- **Bible Translation:** Quote scriptures only from the New World Translation of the Holy Scriptures (Study Edition) unless another version is explicitly requested. When quoting a Bible verse, do it in verbatim, not paraphrased. Cite scriptures with precise verbal formatting, e.g., English: "Matthew chapter 24 verse 14"; Korean: "마태복음 24장 14절".

Scope of Answers:
- If the question goes beyond Bible-related research, kindly but clearly remind the user: "This service is reserved for scripturally-based research using jw.org. Please verify all spiritual information with the official website."

You must maintain a formal and spiritually respectful tone.
"""

LLM_PROMPT_LANGUAGE_RULES = {
    "en": """
Use: Jehovah, not Lord or God
Use: Jesus, not Jesus Christ (unless the original source does)
Use: God’s purpose, not God’s plan
Do not refer to someone as “장로” in conversation—use “형제” or “자매” regardless of position or official appointment.
""",
    "ko": """
예수님 → 예수
하나님 → 하느님
예배/경배 → 숭배
십자가 → 형주
장로님 → 장로
신앙 → 믿음
하느님의 계획 → 하느님의 목적
형제님/자매님 → 형제/자매 only. Do not use honorifics like “님” unless quoting directly.
Do not refer to someone as “장로” in conversation—use “형제” or “자매” regardless of position or official appointment.
""",
    # Add rules for other languages as needed
    "default": """
"""
}

# --- Streamlit Session State Management ---
if "faiss_db" not in st.session_state:
    st.session_state.faiss_db = None
if "faiss_documents" not in st.session_state:
    st.session_state.faiss_documents = []
if "messages" not in st.session_state:
    st.session_state.messages = []
if "llm_chain" not in st.session_state:
    st.session_state.llm_chain = None
if "faiss_loaded" not in st.session_state:
    st.session_state.faiss_loaded = False
if "show_debug" not in st.session_state:
    st.session_state.show_debug = False


# --- Helper Functions ---
def load_and_split_pdfs(uploaded_files):
    """Loads uploaded PDF files and splits them into text chunks using PyMuPDF."""
    documents = []
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)

    for uploaded_file in uploaded_files:
        try:
            with fitz.open(stream=uploaded_file.getvalue(), filetype="pdf") as doc:
                text_with_spaces = ""
                for page_num, page in enumerate(doc):
                    # Use get_text("blocks") for a more structured extraction
                    blocks = page.get_text("blocks")
                    
                    if not blocks:
                        # Fallback to simple text extraction if no blocks are found
                        text_with_spaces += page.get_text() + "\n"
                        continue

                    # Sort blocks by their y-coordinate
                    blocks.sort(key=lambda b: b[1])
                    
                    for _, _, _, _, block_text, _, _ in blocks:
                        # Split by space and then re-join to handle complex spacing
                        cleaned_block_text = " ".join(block_text.split())
                        # Add a space after each block to separate them
                        text_with_spaces += cleaned_block_text + " "
                    text_with_spaces += "\n" # Add a new line for each page

                # Create a single Document object with the extracted and spaced text
                source_document = Document(page_content=text_with_spaces, metadata={"source": uploaded_file.name})
                
                # Split the entire document's text into chunks
                docs = text_splitter.split_documents([source_document])
                
                # Update metadata for each chunk
                for i, doc in enumerate(docs):
                    doc.metadata['source'] = uploaded_file.name
                    doc.metadata['page'] = i + 1  # Using chunk number as a proxy for page
                
                documents.extend(docs)
                
        except Exception as e:
            st.error(f"Error processing {uploaded_file.name}: {e}")
            
    return documents

def create_and_save_faiss_db(documents):
    """Creates a FAISS vector store from documents and saves it locally."""
    with st.spinner("Creating knowledge base..."):
        embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
        
        # Use a dictionary to handle deduplication and replacement
        doc_dict = {}
        
        # Load existing documents into the dictionary
        if st.session_state.faiss_db:
            existing_docs = list(st.session_state.faiss_db.docstore._dict.values())
            for doc in existing_docs:
                key = f"{doc.metadata.get('source')}-{doc.metadata.get('page')}"
                doc_dict[key] = doc
        
        # Add or replace newly uploaded documents
        for doc in documents:
            key = f"{doc.metadata.get('source')}-{doc.metadata.get('page')}"
            doc_dict[key] = doc
            
        all_documents = list(doc_dict.values())
        
        # If the FAISS index directory exists, remove it to start fresh
        if os.path.exists(FAISS_INDEX_PATH):
            shutil.rmtree(FAISS_INDEX_PATH)

        # Generate a unique ID for each document chunk and create a list of IDs
        doc_with_ids = []
        doc_ids = []
        for doc in all_documents:
            unique_id = str(uuid.uuid4())
            doc.metadata['uuid'] = unique_id
            doc_ids.append(unique_id)
            doc_with_ids.append(doc)
            
        st.session_state.faiss_db = FAISS.from_documents(doc_with_ids, embeddings, ids=doc_ids)
        st.session_state.faiss_db.save_local(FAISS_INDEX_PATH)
        
        # Corrected logic to ensure no duplicates in the document list
        st.session_state.faiss_documents = sorted(list(set(doc.metadata.get('source') for doc in all_documents)))
        
        st.session_state.faiss_loaded = True
    st.success("Knowledge base updated!")
    st.rerun()

def load_faiss_db():
    """Loads a FAISS vector store from a local file if it exists."""
    if os.path.exists(FAISS_INDEX_PATH):
        try:
            with st.spinner("Loading knowledge base..."):
                embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
                st.session_state.faiss_db = FAISS.load_local(
                    FAISS_INDEX_PATH,
                    embeddings,
                    allow_dangerous_deserialization=True  # Required for security in some versions
                )
                
                # Update the list of loaded documents from the vector store metadata
                docs = st.session_state.faiss_db.docstore._dict.values()
                st.session_state.faiss_documents = sorted(list(set([doc.metadata.get('source', 'Unknown Document') for doc in docs])))
                st.session_state.faiss_loaded = True
            st.success("Knowledge base loaded! ")
        except Exception as e:
            st.error(f"Error loading FAISS database: {e}. Please clear the knowledge base and try again.")
            st.session_state.faiss_db = None
            st.session_state.faiss_loaded = False
    else:
        st.session_state.faiss_db = None
        st.session_state.faiss_loaded = False

def clear_faiss_db():
    """Clears the local FAISS vector store and session state."""
    if os.path.exists(FAISS_INDEX_PATH):
        try:
            import shutil
            shutil.rmtree(FAISS_INDEX_PATH)
        except Exception as e:
            st.error(f"Error clearing knowledge base: {e}")
    
    # Always clear the session state regardless of file system status
    st.session_state.faiss_db = None
    st.session_state.faiss_documents = []
    st.session_state.messages = []
    st.session_state.faiss_loaded = False
    st.success("Knowledge base cleared!")
    st.rerun()

def detect_language(text):
    """Detects the language of the input text based on character ranges."""
    # Check for Korean (Hangul)
    if re.search(r'[\uac00-\ud7a3]', text):
        return 'ko'
    # Check for Japanese (Hiragana, Katakana, Kanji)
    if re.search(r'[\u3040-\u30ff\u3400-\u4dbf\u4e00-\u9faf]', text):
        return 'ja'
    # Check for Chinese (Hanzi)
    if re.search(r'[\u4e00-\u9fff]', text):
        return 'zh'
    # Check for Spanish (common accented characters)
    if re.search(r'[áéíóúÁÉÍÓÚñÑ]', text):
        return 'es'
    # Default to English
    return 'en'


# --- Main Streamlit App Layout ---
def main():
    st.set_page_config(page_title=APP_TITLE, layout="wide")
    st.title(APP_TITLE)

    # Load FAISS database once at the start of the session
    if not st.session_state.faiss_loaded:
        load_faiss_db()

    # Sidebar for document management
    with st.sidebar:
        st.header("Document Management")
        st.session_state.show_debug = st.checkbox("Show Debugging Information")
        uploaded_files = st.file_uploader(
            "Upload PDFs to add to the knowledge base",
            type="pdf",
            accept_multiple_files=True
        )

        # Handle document upload
        if st.button("Add to Knowledge Base"):
            if uploaded_files:
                with st.spinner("Processing documents..."):
                    documents = load_and_split_pdfs(uploaded_files)
                    create_and_save_faiss_db(documents)
            else:
                st.warning("Please upload at least one PDF.")
        
        # Display list of loaded documents
        st.subheader("Loaded Documents")
        if st.session_state.faiss_documents:
            # Use st.expander for a long list of documents
            with st.expander("Show/Hide Document List"):
                st.info(f"Currently loaded documents: {len(st.session_state.faiss_documents)}")
                for doc_name in st.session_state.faiss_documents:
                    st.write(f"- {doc_name}")
        else:
            st.info("No documents loaded.")
            
        # Option to clear the database
        if st.button("Clear Knowledge Base"):
            clear_faiss_db()

    # Main chat interface
    st.subheader("Conversational Q&A with your documents")
    
    # Display chat messages from history
    for msg in st.session_state.messages:
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"])
            
    # Process user input
    if user_prompt := st.chat_input("Ask a question about your documents..."):
        if not st.session_state.faiss_db:
            st.error("No documents in the knowledge base. Please upload some PDFs first.")
            st.session_state.messages.append({"role": "user", "content": user_prompt})
            st.session_state.messages.append({"role": "assistant", "content": "No documents in the knowledge base. Please upload some PDFs first."})
            st.rerun()
            return
            
        # Add user message to chat history
        st.session_state.messages.append({"role": "user", "content": user_prompt})
        with st.chat_message("user"):
            st.markdown(user_prompt)

        # Detect the user's language and get the corresponding rules
        detected_lang = detect_language(user_prompt)
        lang_rules = LLM_PROMPT_LANGUAGE_RULES.get(detected_lang, LLM_PROMPT_LANGUAGE_RULES['default'])
        
        # Construct the dynamic system prompt
        dynamic_system_prompt = LLM_SYSTEM_PROMPT_BASE + lang_rules
        
        # Initialize LLM chain with the dynamic prompt
        try:
            chat_model = ChatOllama(model="llama3:latest")
            prompt = ChatPromptTemplate.from_messages([
                ("system", dynamic_system_prompt + "\n\nContext:\n{context}"),
                ("placeholder", "{chat_history}"),
                ("human", "Question: {input}"),
                ("human", f"**IMPORTANT**: Provide your final answer in the same language as the user's question, which is {detected_lang}. Do not translate citations or proper nouns from the original documents.")
            ])
            st.session_state.llm_chain = create_stuff_documents_chain(chat_model, prompt)
        except Exception as e:
            st.error(f"Error initializing LLM: {e}. Please ensure Ollama is running and Llama3 is downloaded. You can download it by running `ollama pull llama3:latest` in your terminal.")
            return

        # Get response from the RAG chain
        with st.chat_message("assistant"):
            with st.spinner("Searching and generating response..."):
                retriever = st.session_state.faiss_db.as_retriever()
                retrieval_chain = create_retrieval_chain(retriever, st.session_state.llm_chain)

                # The LangChain expression language allows for a simple invocation
                response = retrieval_chain.invoke({
                    "chat_history": st.session_state.messages,
                    "input": user_prompt
                })
                
                # Get the generated answer
                ai_response_content = response['answer']

                # Display debugging information if enabled
                if st.session_state.show_debug:
                    with st.expander("Debugging Information"):
                        st.subheader("Vector Store Search Results")
                        for i, doc in enumerate(response['context']):
                            st.markdown(f"**Document {i+1}:** {doc.metadata.get('source', 'Unknown')} (page {doc.metadata.get('page', 'Unknown')})")
                            st.code(doc.page_content, language="text")
                        
                        st.subheader("LLM Raw Output")
                        st.code(ai_response_content, language="text")

                # Display the final response
                st.markdown(ai_response_content)
                
                # Add source information
                if 'context' in response and response['context']:
                    st.markdown("---")  # Add a horizontal line for separation
                    st.markdown("**Sources:**")
                    sources = sorted(list(set(f"- {doc.metadata.get('source', 'Unknown Document')}, page {doc.metadata.get('page', 'Unknown Page')}" for doc in response['context'])))
                    for source in sources:
                        st.markdown(source)

                # Add assistant message to chat history
                st.session_state.messages.append({"role": "assistant", "content": ai_response_content})

if __name__ == "__main__":
    main()
