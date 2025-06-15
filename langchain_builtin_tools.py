from langchain_community.tools import DuckDuckGoSearchRun, WikipediaQueryRun
from langchain_community.utilities import WikipediaAPIWrapper

# search_tool = DuckDuckGoSearchRun()

# result = search_tool.run("AIR INDIA")
# print(result)
# print("\nSearch Tool Details:")
# print(f"{search_tool.name}\n")
# print(f"{search_tool.description}\n")
# print(f"{search_tool.args}\n")


# Wikipedia = WikipediaQueryRun(api_wrapper=WikipediaAPIWrapper())

# print(Wikipedia.name)
# print(Wikipedia.description)
# print(Wikipedia.args)
# print(Wikipedia.run("HUNTER X HUNTER"))

import os
from langchain_community.tools import tool
from langchain_openai import AzureChatOpenAI
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain.agents import create_openai_functions_agent, AgentExecutor

api_base = os.getenv("AZURE_OPENAI_API_BASE")
api_version = os.getenv("AZURE_OPENAI_API_VERSION")
api_key = os.getenv("AZURE_API_KEY")
deployment_name = os.getenv("AZURE_OPENAI_API_NAME")

llm=AzureChatOpenAI(
    azure_endpoint=api_base,
    openai_api_version=api_version,
    openai_api_key=api_key,
    deployment_name=deployment_name,
    model_name="gpt-4o",
    temperature=0.9,
    max_tokens=100,
)  # type: ignore

@tool
def multiply(a:int, b:int) -> int:
    """Multiply two numbers."""
    return a*b

mul_tool = multiply.run({"a": 2, "b": 3})

@tool
def add(a:int, b:int) -> int:
    """Add Two numbers."""
    return a+b

tool = [multiply, add]

prompt = ChatPromptTemplate.from_messages(
    [
        ("system", "You are a smart assistant who uses tools to answer math questions."),
        ("user", "{input}"),
        MessagesPlaceholder(variable_name="agent_scratchpad")
    ]
)

agent = create_openai_functions_agent(llm=llm, tools=tool, prompt=prompt)

agent_executor = AgentExecutor(agent=agent, tools=tool)

response = agent_executor.invoke({"input": "add 2 and 3 then multiply result by 3"})
print("Response: ", response["output"])

# llm_with_tools = llm.bind_tools([multiply, add])
# result = llm_with_tools.invoke("Multiply 2 with 3")
# print(result)
