from langgraph.graph import MessagesState, StateGraph, END, START
from langchain_core.messages import HumanMessage, SystemMessage, AnyMessage
from langchain_community.tools.yahoo_finance_news import YahooFinanceNewsTool
from langchain_community.tools import DuckDuckGoSearchRun
from langgraph.prebuilt import tools_condition, create_react_agent, ToolNode
from langchain_community.tools import tool
from IPython.display import Image, display
from typing import Annotated, TypedDict
import operator
from langgraph.graph.message import add_messages
import os
from langchain_openai import AzureChatOpenAI

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
)  # type:ignore

search = DuckDuckGoSearchRun()

finance = YahooFinanceNewsTool()

system_message = SystemMessage(
    content = (
        "You are a professional stock market analyst. Your task is to determine whether it is currently a good time to invest in a given stock. "
        "Base your analysis strictly on the most recent and reliable financial news, market trends, and stock data. "
        "Do not respond to queries unrelated to the stock market. "
        "After thorough research, provide a clear 'Yes' or 'No' recommendation. "
        "Include a brief two-line justification based on current news or market data. "
        "Also, clearly mention the date of the news or data you used to make this decision, and provide the latest available stock price at the end."
    )
)
tools = [search, finance]
llm_with_tools = llm.bind_tools(tools)

def reasoner(state: MessagesState):
    return {"messages": [llm_with_tools.invoke([system_message] + state["messages"])]}

# Graph
builder = StateGraph(MessagesState)

# Add Nodes
builder.add_node("reasoner", reasoner)
builder.add_node("tools", ToolNode(tools))

# Add Edges
builder.add_edge(START, "reasoner")

builder.add_conditional_edges(
    "reasoner",
    tools_condition
)

builder.add_edge("tools", "reasoner")

react_graph = builder.compile()

messages = [HumanMessage(content="Whether should i invest in apple stock? and what is the current price of apple stock?")]
messages = react_graph.invoke({"messages": messages})

for m in messages["messages"]:
    m.pretty_print()
