import PyPDF2
import streamlit as st
import os
import re
from langchain.chat_models import AzureChatOpenAI
from langchain.prompts import PromptTemplate
from dotenv import load_dotenv

def extract_pdf(pdf_file):
    """Extract text from PDF file."""
    text = ""
    pdf = PyPDF2.PdfReader(pdf_file)
    for page in pdf.pages:
        text += page.extract_text()
    return text

def extract_score(text):
    """Extract first integer score from LLM output safely."""
    text = text.replace("*", "")  # Remove asterisks first
    match = re.search(r"Score:\s*(\d+)", text)
    if match:
        return int(match.group(1))
    else:
        return 0  # fallback if not found
    
def extract_explanation(text):
    """Extract explanation from LLM output safely."""
    text = text.replace("*", "")  # Remove asterisks first
    match = re.search(r"Explanation:\s*(.*)", text, re.DOTALL)
    if match:
        return match.group(1).strip()
    return "Not available"

load_dotenv()

openai_endpoint = os.getenv("AZURE_OPENAI_API_BASE")
openai_key = os.getenv("AZURE_API_KEY")
openai_version = os.getenv("AZURE_OPENAI_API_VERSION")
deployment = os.getenv("AZURE_OPENAI_API_NAME")

def score_resume(job_description, resume):
    """Send to AzureChatOpenAI and get score back."""
    prompt = PromptTemplate.from_template("""
    You are a recruiter. I'll give you a Job Description and a Resume. 
    Analyze the Resume against the Job Description and score it from 0-100 based on how well it fits the job criteria. 
    Also provide a brief explanation for your score.

    Job Description:
    {job_description}

    Resume:
    {resume}

    Your response should be in this format:

    Score: <score>
    Explanation: <1-2 sentence explanation>
    """)
    prompt = prompt.format(job_description=job_description, resume=resume)

    llm = AzureChatOpenAI(
        azure_endpoint=openai_endpoint,
        azure_deployment=deployment,
        openai_api_key=openai_key,
        openai_api_version=openai_version,
        temperature=0,
        max_tokens=200
    )  # type:ignore
    result = llm.predict(prompt)
    return result


def main():
    st.title("Resume Ranker")
    st.text("Compare resumes against a job description.")
    
    job_file = st.file_uploader("Job Description (txt)", type=['txt'])
    resumes = st.file_uploader("Resume PDFs", type=['pdf'], accept_multiple_files=True)

    if job_file and resumes:
        job_description = job_file.read().decode()
        scores = []

        for res_file in resumes:
            res_txt = extract_pdf(res_file)
            result = score_resume(job_description, res_txt)
            score = extract_score(result)
            explanation = extract_explanation(result)
            scores.append((res_file.name, score, explanation))

        scores_sorted = sorted(scores, key=lambda x: x[1], reverse=True)

        st.write("**Ranking:**")
        for i, (filename, score, explanation) in enumerate(scores_sorted, 1):
            st.write(f"{i}. {filename} - Score: {score}")
            st.write(f"Explanation: {explanation}")
            st.write("----------------------------")

if __name__ == "__main__":
    main()
