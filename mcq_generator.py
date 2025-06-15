import streamlit as st
from PyPDF2 import PdfReader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.chat_models import AzureChatOpenAI
from langchain.prompts import PromptTemplate
from langchain.schema.messages import HumanMessage
from langchain.embeddings import AzureOpenAIEmbeddings
from langchain.vectorstores import FAISS
import os
from dotenv import load_dotenv

load_dotenv()


def extract_pdf(pdf_file):
    """Extract text from PDF file."""
    text = ""
    pdf = PdfReader(pdf_file)
    for page in pdf.pages:
        text += page.extract_text()
    return text


def chunk(text, chunk_size=500, chunk_overlap=50):
    """Split text into chunks."""
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap
    )
    return splitter.split_text(text)


def create_vectordb(chunks, embedding):
    """Create vector database from text chunks."""
    return FAISS.from_texts(chunks, embedding)


def generate_questions(pdf_file):
    """Main pipeline to generate questions from PDF."""
    # Azure Embedding
    embedding = AzureOpenAIEmbeddings(
        azure_endpoint=os.getenv("TEXT_EMBEDDING_API_BASE"),
        openai_api_key=os.getenv("AZURE_API_KEY"),
        deployment=os.getenv("TEXT_EMBEDDING_API_NAME"),
        chunk_size=1000,
    )
    # Extract, Chunk and Vectorise
    text = extract_pdf(pdf_file)
    chunks = chunk(text)
    vectordb = create_vectordb(chunks, embedding)

    retriever = vectordb.as_retriever()
    docs = retriever.get_relevant_documents("Generate questions.")
    context = "\n\n".join([d.page_content for d in docs])

    # Prepare prompt
    question_prompt = PromptTemplate.from_template("""
Using the following context:

{context}

Generate 10 multiple-choice questions with 4 options each and answers.

Please respond in a clear format.

    """)

    input_prompt = question_prompt.format(context=context)

    # Prepare LLM
    llm = AzureChatOpenAI(
        openai_api_base=os.getenv("AZURE_OPENAI_API_BASE"),
        openai_api_version=os.getenv("AZURE_OPENAI_API_VERSION"),
        openai_api_key=os.getenv("AZURE_API_KEY"),
        deployment_name=os.getenv("AZURE_OPENAI_API_NAME"),
        model_name="gpt-4o",
        temperature=0.7,
        top_p=0.9,
    )  # type: ignore

    result = llm.generate([[HumanMessage(content=input_prompt)]])

    return result.generations[0][0].text


def main():
    """Streamlit UI Main function."""
    st.title("Question Paper Generator")
    st.write("➥ Upload a PDF and generate questions from its content.")
    uploaded_file = st.file_uploader("Choose a PDF file", type=["pdf"])

    if uploaded_file:
        with st.spinner("Generating questions..."):
            questions = generate_questions(uploaded_file)

        st.success("Questions generated successfully.")

        # Split questions by lines and format nicely
        lines = questions.splitlines()
        question_count = 1
        for line in lines:
            if line.startswith(str(question_count) + "."):
                st.markdown(f"**{line}**")
                question_count += 1
            elif line.startswith(("-", "*", "A)", "B)", "C)", "D)")):
                st.write(line)
            else:
                st.text(line)



if __name__ == "__main__":
    main()
