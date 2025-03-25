import streamlit as st
import os
import time
from dotenv import load_dotenv

# LangChain & NVIDIA Imports
from langchain_nvidia_ai_endpoints import NVIDIAEmbeddings, ChatNVIDIA
from langchain_community.document_loaders import PyPDFDirectoryLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_core.prompts import ChatPromptTemplate
from langchain.chains import create_retrieval_chain
from langchain_community.vectorstores import FAISS

# ✅ Load environment variables
load_dotenv()

# ✅ Fetch API key from Streamlit Secrets (Required for Cloud Deployment)
if "NVIDIA_API_KEY" in st.secrets:
    os.environ["NVIDIA_API_KEY"] = st.secrets["NVIDIA_API_KEY"]
else:
    st.error("❌ NVIDIA API Key is missing! Add it in Streamlit Secrets.")
    st.stop()

# Streamlit UI
st.title("🔍 Nvidia NIM: US Census AI Assistant")
st.subheader("📚 RAG App for US Census Data")

# ✅ Function to Create Vector Embeddings
def vector_embedding():
    if "vectors" not in st.session_state:
        st.write("🔄 Processing documents for embeddings...")

        try:
            # Initialize NVIDIA embeddings
            st.session_state.embeddings = NVIDIAEmbeddings()

            # Load PDFs
            st.session_state.loader = PyPDFDirectoryLoader("./us_census")
            st.session_state.docs = st.session_state.loader.load()

            if not st.session_state.docs:
                st.error("⚠️ No documents found! Please check the './us_census' directory.")
                return

            # Split documents into chunks
            st.session_state.text_splitter = RecursiveCharacterTextSplitter(chunk_size=700, chunk_overlap=50)
            st.session_state.final_documents = st.session_state.text_splitter.split_documents(st.session_state.docs[:30])

            # Create vector store using FAISS
            st.session_state.vectors = FAISS.from_documents(st.session_state.final_documents, st.session_state.embeddings)

            st.success("✅ Vector Store DB is Ready!")

        except Exception as e:
            st.error(f"⚠️ Error in vector embedding: {str(e)}")

# Initialize LLM
llm = ChatNVIDIA(model="meta/llama-3.3-70b-instruct")

# Define Prompt
prompt = ChatPromptTemplate.from_template(
    """
    Answer the questions based on the provided context only.
    Provide the most accurate response.
    <context>
    {context}
    </context>
    Question: {input}
    """
)

# User Input
prompt1 = st.text_input("📝 Enter Your Question from Documents", placeholder="What is the job status in US?")

# Button to Generate Embeddings
if st.button("📚 Documents Embedding"):
    vector_embedding()

# ✅ Ensure Vectors Exist Before Processing Query
if prompt1:
    if "vectors" not in st.session_state:
        st.warning("⚠️ Please click 'Documents Embedding' first to generate vectors.")
    else:
        try:
            # Create Document Chain
            document_chain = create_stuff_documents_chain(llm, prompt)
            retriever = st.session_state.vectors.as_retriever()
            retrieval_chain = create_retrieval_chain(retriever, document_chain)

            # Start Processing
            start = time.process_time()
            response = retrieval_chain.invoke({'input': prompt1})
            st.write(f"⏳ Response time: {time.process_time() - start:.2f} seconds")

            # Display Response
            st.write("### 🤖 Answer:")
            st.write(response['answer'])

            # Expand for Similar Documents
            with st.expander("📄 Document Similarity Search"):
                for i, doc in enumerate(response["context"]):
                    st.write(doc.page_content)
                    st.write("🔹" * 20)

        except Exception as e:
            st.error(f"⚠️ Error in processing query: {str(e)}")
