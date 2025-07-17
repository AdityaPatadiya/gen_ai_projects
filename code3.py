from langchain_core.prompts import PromptTemplate, ChatPromptTemplate
from langchain_core.messages import HumanMessage, AIMessage, SystemMessage
from langchain_openai import AzureChatOpenAI
from langchain.chains import LLMChain
import os
from dotenv import load_dotenv

load_dotenv()

llm = AzureChatOpenAI(
    azure_endpoint=os.getenv("AZURE_OPENAI_API_BASE"),
    openai_api_version=os.getenv("AZURE_OPENAI_API_VERSION"),
    openai_api_key=os.getenv("AZURE_API_KEY"),
    deployment_name=os.getenv("AZURE_OPENAI_API_NAME"),
    model_name="gpt-4o",
    temperature=0.7,
)  # type: ignore

my_prompt = ChatPromptTemplate.from_messages(
    [
        SystemMessage(
            content="""You are an expert in programming languages.
            You help users understand and explore different programming languages.
            Only answer questions related to programming languages.
            If the question is outside that scope, reply with "I don't know"."""
        ),
        ("human", "Explain the {topic} programming language in a {selected} manner."),
    ]
)

chain = my_prompt | llm


result = chain.invoke({
    "topic": "Python",
    "selected": "concise",
})

print(result.content)
