from typing import TypedDict, Annotated, List
from langchain_core.messages import BaseMessage, HumanMessage

# Import AgentState to ensure type hints are consistent
from core.state import AgentState # Notice the import: from package.module import Class

def route_agent(state: AgentState) -> str:
    """
    Routes the incoming message to the appropriate specialized agent.
    
    Args:
        state (AgentState): The current state of the conversation.

    Returns:
        str: The name of the next agent (node) to execute ('clinical_assistant',
             'leave_scheduling', or 'fallback').
    """
    print("---ROUTER AGENT: Analyzing message for routing---")
    messages = state['messages']
    
    # Get the content of the last human message
    last_human_message = ""
    for msg in reversed(messages):
        if isinstance(msg, HumanMessage):
            last_human_message = msg.content.lower()
            break
    
    if not last_human_message:
        print("Router: No human message found, defaulting to fallback.")
        return "fallback"

    # Keywords for Clinical Assistant
    clinical_keywords = ["health", "medical", "symptom", "doctor", "clinic", "fever", "pain", "cold", "headache", "ill"]
    if any(keyword in last_human_message for keyword in clinical_keywords):
        print("Router: Detected clinical query, routing to clinical_assistant.")
        return "clinical_assistant"

    # Keywords for Leave Scheduling Agent
    leave_keywords = ["leave", "vacation", "sick day", "holiday", "absence", "time off", "request", "balance", "apply for"]
    if any(keyword in last_human_message for keyword in leave_keywords):
        print("Router: Detected leave query, routing to leave_scheduling.")
        return "leave_scheduling"

    print("Router: No specific agent detected, routing to fallback.")
    return "fallback"
