import streamlit as st
from PyPDF2 import PdfReader
from fpdf import FPDF
import unicodedata
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


def generate_questions_from_text(text):
    """Generate questions directly from combined text of all PDFs."""
    embedding = AzureOpenAIEmbeddings(
        azure_endpoint=os.getenv("TEXT_EMBEDDING_API_BASE"),
        openai_api_key=os.getenv("AZURE_API_KEY"),
        deployment=os.getenv("TEXT_EMBEDDING_API_NAME"),
        chunk_size=1000,
    )
    chunks = chunk(text)
    vectordb = create_vectordb(chunks, embedding)

    retriever = vectordb.as_retriever()
    docs = retriever.get_relevant_documents("Generate questions.")
    context = "\n\n".join([d.page_content for d in docs])

    question_prompt = PromptTemplate.from_template("""
Using the following context:

{context}

Generate 5 multiple-choice questions with 4 options each and answers.
and also generate 5 short answer questions based on the context.

Please respond in a clear format.

    """)

    input_prompt = question_prompt.format(context=context)

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


def save_to_pdf(questions, output_file):
    """Save questions to PDF file."""
    pdf = FPDF()
    pdf.add_page()
    pdf.set_font("Arial", size = 14)

    for line in questions.splitlines():
        ascii_line = unicodedata.normalize('NFKD', line).encode('ascii', 'ignore').decode()
        pdf.multi_cell(200, 10, ascii_line)

    pdf.output(output_file)


def main():
    """Streamlit UI Main function."""
    st.title("Question Paper Generator")
    st.write("➥ Upload PDF files and generate questions from their content.")
    uploaded_files = st.file_uploader("Choose PDF files", type=["pdf"], accept_multiple_files=True)

    if uploaded_files:
        with st.spinner("Generating questions from documents..."):
            all_text = ""
            for uploaded_file in uploaded_files:
                all_text += extract_pdf(uploaded_file)

            questions = generate_questions_from_text(all_text)

        st.divider()
        st.success("Questions generated successfully.")

        lines = questions.splitlines()
        for line in lines:
            if line.startswith(("-", "*", "A)", "B)", "C)", "D)")):
                st.write(line)
            else:
                st.text(line)

        if st.button("Save questions to PDF"):
            pdf_file = "generated_questions.pdf"
            save_to_pdf(questions, pdf_file)

            with open(pdf_file, "rb") as file:
                st.download_button(
                    label="Download PDF",
                    data=file,
                    file_name=pdf_file,
                    mime="application/pdf"
                )


if __name__ == "__main__":
    main()
