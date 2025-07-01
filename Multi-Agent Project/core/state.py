from typing import TypedDict, Annotated, List
from langchain_core.messages import BaseMessage
import operator

# Define the AgentState for LangGraph
class AgentState(TypedDict):
    """
    Represents the state of the multi-agent conversation.
    Messages are accumulated throughout the conversation.
    """
    messages: Annotated[List[BaseMessage], operator.add]
    # You can add other state variables here if needed for more complex logic
    # For example:
    # current_task: str
    # extracted_entities: dict
