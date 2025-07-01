import os
from typing import List, Annotated, TypedDict
from langchain_core.messages import BaseMessage, HumanMessage, AIMessage
from langchain_openai import AzureChatOpenAI
from langchain.prompts import ChatPromptTemplate
from langchain_core.tools import tool # Import tool decorator

# Define the shared state (imported from core.state)
from core.state import AgentState # Notice the import: from package.module import Class

# Initialize LLM (ensure OPENAI_API_KEY is set in your environment or .env)
llm = AzureChatOpenAI(
    
    model="gpt-4o", 
    temperature=0.7
    )

# --- Define Tools for Clinical Assistant ---
@tool
def get_medical_guideline(disease: str) -> str:
    """
    Retrieves general medical guidelines for a given disease or symptom.
    Args:
        disease (str): The disease or symptom to get guidelines for.
    Returns:
        str: General medical advice.
    """
    print(f"---TOOL CALL: get_medical_guideline for {disease}---")
    # This is a dummy implementation. In a real system, it would query a medical database.
    disease = disease.lower()
    if "fever" in disease:
        return "For fever, ensure hydration, rest, and consider over-the-counter fever reducers like paracetamol. Consult a doctor if symptoms worsen or persist or if fever is high/long-lasting."
    elif "cold" in disease:
        return "For common cold, rest, fluids, and symptom relievers (like decongestants). Not a serious condition, but consult a doctor if severe."
    elif "headache" in disease:
        return "For headaches, rest, hydration, and pain relievers. Seek medical attention if severe, sudden, or accompanied by other neurological symptoms."
    else:
        return f"I can provide general advice, but for '{disease}', it's best to consult a medical professional for personalized guidance."

# --- Clinical Assistant Agent Node ---
def clinical_assistant_agent(state: AgentState):
    """
    Processes health-related queries using an LLM and potentially medical tools.
    """
    print("---CLINICAL ASSISTANT AGENT PROCESSING---")
    messages = state['messages']
    last_human_message = ""
    if messages and isinstance(messages[-1], HumanMessage):
        content = messages[-1].content
        if isinstance(content, list):
            last_human_message = " ".join(
                [c if isinstance(c, str) else str(c) for c in content]
            )
        else:
            last_human_message = str(content)

    # Define the system prompt for the clinical assistant
    system_prompt = (
        "You are a helpful clinical assistant. Your primary goal is to provide general, non-diagnostic "
        "health information and suggest consulting a medical professional for any health concerns or "
        "personalized advice. You can use provided tools to assist."
    )

    # Simplified tool invocation logic for demonstration
    response_content = ""
    if "guideline" in last_human_message.lower() or "advice on" in last_human_message.lower():
        # Attempt to extract disease/symptom for the tool
        # A more robust solution would use LLM for structured extraction
        disease_match = last_human_message.lower().replace("what are the guidelines for", "").replace("give me advice on", "").strip()
        if disease_match:
            tool_result = get_medical_guideline(disease_match)
            response_content = f"Here is some general information: {tool_result}"
        else:
            response_content = "Please specify the disease or symptom you need guidelines for."
    else:
        # If no specific tool is triggered, just use the LLM for general response
        prompt = ChatPromptTemplate.from_messages([
            {"role": "system", "content": system_prompt},
            {"role": "human", "content": last_human_message}
        ])
        # Format the prompt to get the message sequence
        formatted_messages = prompt.format_messages()
        llm_response = llm.invoke(formatted_messages)
        response_content = llm_response.content

    return {"messages": [AIMessage(content=response_content)]}
