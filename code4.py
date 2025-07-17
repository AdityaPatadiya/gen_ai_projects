from langchain_core.prompts import PromptTemplate, ChatPromptTemplate
from langchain.prompts import FewShotPromptTemplate
from langchain_core.messages import HumanMessage, AIMessage, SystemMessage
from langchain_openai import AzureChatOpenAI
from langchain.chains import LLMChain
import os
from dotenv import load_dotenv

load_dotenv()

api_base = os.getenv("AZURE_OPENAI_API_BASE")
api_version = os.getenv("AZURE_OPENAI_API_VERSION")
api_key = os.getenv("AZURE_API_KEY")
deployment_name = os.getenv("AZURE_OPENAI_API_NAME")

llm = AzureChatOpenAI(
    azure_endpoint=api_base,
    openai_api_version=api_version,
    openai_api_key=api_key,
    deployment_name=deployment_name,
    model_name="gpt-4o",
    temperature=0.7,
    top_p=0.9,
    max_tokens=100,
)  # type: ignore

examples = [
    {"question": "What is the capital of France?", "answer": "The capital of France is Paris."},
    {"question": "What is the capital of Germany?", "answer": "The capital of Germany is Berlin."},
    {"question": "What is the capital of Italy?", "answer": "The capital of Italy is Rome."},
]

examples_prompt = PromptTemplate(
    input_variables=["question", "answer"],
    template="Question: {question}\nAnswer: {answer}",
)

few_shot_prompt = FewShotPromptTemplate(
    examples=examples,
    example_prompt=examples_prompt,
    input_variables=["question"],
    suffix="Question: {question}\nAnswer:",
    prefix="You are a helpful assistant. Answer the following question."
)

# formatted_prompt = few_shot_prompt.format(question="What is the capital of Spain?")
# print(formatted_prompt)

llmchain = LLMChain(
    llm=llm,
    prompt=few_shot_prompt,
)

result = llmchain.invoke({"question": "What is the capital of India?"})
# print(formatted_prompt)
print(result['text'])
