# ReAct Agent with Context

from langgraph.graph import MessagesState, StateGraph, START
from langchain_core.messages import HumanMessage, SystemMessage, AnyMessage
from langchain_community.tools import DuckDuckGoSearchRun
from langgraph.prebuilt import tools_condition
from langgraph.prebuilt import ToolNode
from langchain_community.tools import tool
import os
from langchain_openai import AzureChatOpenAI

api_base = os.getenv("AZURE_OPENAI_API_BASE")
api_version = os.getenv("AZURE_OPENAI_API_VERSION")
api_key = os.getenv("AZURE_API_KEY")
# deployment_name = os.getenv("AZURE_OPENAI_API_NAME")

llm = AzureChatOpenAI(
    azure_endpoint=api_base,
    openai_api_version=api_version,
    openai_api_key=api_key,
    deployment_name=deployment_name,
    model_name="gpt-4o",
    temperature=0.7,
    top_p=0.9,
    max_tokens=100,
)  # type:ignore

response = llm.predict(
    "Who is the current prime minister of the India?",
)

print(response)
print("-----------------------------")

@tool
def add(a:int, b:int) -> int:
    """Add two numbers."""
    return a + b

@tool
def multiply(a:int, b:int) -> int:
    """Multiply two numbers."""
    return a * b

@tool
def divide(a:int, b:int) -> float:
    """Divide two numbers."""
    return a / b

search = DuckDuckGoSearchRun()
search.invoke("Who is a current president of USA?")

tools = [add, multiply, divide, search]

llm_with_tools = llm.bind_tools(tools)

sys_msg = SystemMessage(content="You are a helpful assistant tasked that can perform calculation and search the web.")

def resoner(state: MessagesState):
    return {"messages": [llm_with_tools.invoke([sys_msg] + state["messages"])]}

# Graph
builder = StateGraph(MessagesState)

# Add notes
builder.add_node("resoner", resoner)
builder.add_node("tools", ToolNode(tools))  # for the tools

# Add edges
builder.add_edge(START, "resoner")

builder.add_conditional_edges(
    "resoner",
    tools_condition,
)

builder.add_edge("tools", "resoner")
react_graph = builder.compile()

# display(Image(react_graph.get_graph(xray=True).draw_mermaid_png()))

png_bytes = react_graph.get_graph(xray=True).draw_mermaid_png()

# Save to file
with open("files/langgraph_structure.png", "wb") as f:
    f.write(png_bytes)

messages = [HumanMessage(content="What is 2 times of virat kohli's age?")]
print(messages)

messages = react_graph.invoke({"messages": messages})

print("------------------START---------------------------------")
print(messages)

for m in messages["messages"]:
    m.pretty_print()

print("------------------END---------------------------------")
