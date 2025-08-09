from langchain_community.document_loaders import TextLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.chains import RetrievalQA
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_openai import AzureChatOpenAI
import os
import streamlit as st

# Set credentials directly (REPLACE with your actual values)
os.environ["AZURE_OPENAI_API_KEY"] = "<replace_with_your_api_key>"
os.environ["AZURE_OPENAI_API_BASE"] = "<replace_with_your_api_base_url>"
os.environ["AZURE_OPENAI_API_VERSION"] = "2024-12-01-preview"
os.environ["AZURE_OPENAI_API_NAME"] = "gpt-4o"

# Load credentials
api_base = os.getenv("AZURE_OPENAI_API_BASE")
api_key = os.getenv("AZURE_OPENAI_API_KEY")
api_version = os.getenv("AZURE_OPENAI_API_VERSION")
deployment_name = os.getenv("AZURE_OPENAI_API_NAME")


# Embeddings
embedding_model = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")

# Load and split documents
loader = TextLoader("marwadi_university.txt")
documents = loader.load()
text_splitter = RecursiveCharacterTextSplitter(chunk_size=100, chunk_overlap=20)
split_docs = text_splitter.split_documents(documents)
print(f"Number of split documents: {len(split_docs)}")

# Vector store and retriever
vector_db = FAISS.from_documents(split_docs, embedding_model)
retriever = vector_db.as_retriever()

# Azure LLM
llm = AzureChatOpenAI(
    azure_endpoint=api_base,
    api_key=api_key,
    api_version=api_version,
    deployment_name=deployment_name,
    model_name="gpt-4o",
    max_tokens=300,
)

# QA chain
qa_chain = RetrievalQA.from_chain_type(
    llm=llm,
    retriever=retriever,
)

# Run loop
while True:
    query = input("Ask a question (or 'exit' to quit): ")
    if query.lower() == 'exit':
        break
    response = qa_chain.run(query)
    print(f"Response: {response}")
    print("-----------------------------------------------------------------------------")
