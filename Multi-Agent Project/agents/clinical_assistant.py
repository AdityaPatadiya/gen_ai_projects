import os
import requests
from typing import List, Annotated, TypedDict
from langchain_core.messages import BaseMessage, HumanMessage, AIMessage
from langchain_openai import AzureChatOpenAI
from langchain.prompts import ChatPromptTemplate
from langchain_core.tools import tool # Import tool decorator
from dotenv import load_dotenv

# Define the shared state (imported from core.state)
from core.state import AgentState # Notice the import: from package.module import Class

load_dotenv()

# Initialize LLM (ensure OPENAI_API_KEY is set in your environment or .env)
llm = AzureChatOpenAI(
    azure_endpoint=os.getenv("AZURE_OPENAI_API_BASE"),
    api_key=os.getenv("AZURE_API_KEY"),
    api_version=os.getenv("AZURE_OPENAI_API_VERSION"),
    deployment_name=os.getenv("AZURE_OPENAI_API_NAME"),
    model_name="gpt-4o",
    max_tokens=300,
) # type: ignore

# --- Define Tools for Clinical Assistant ---
@tool
def get_medical_guideline(disease: str) -> str:
    """
    Retrieves general medical guidelines for a given disease or symptom from a hypothetical API.
    Args:
        disease (str): The disease or symptom to get guidelines for.
    Returns:
        str: General medical advice.
    """
    print(f"---TOOL CALL: get_medical_guideline for {disease}---")
    # This is a hypothetical API call. In reality, you'd integrate with a real medical API.
    # Example: A simplified endpoint that returns structured data.
    API_URL = "https://api.example.com/medical_guidelines" # REPLACE with a real API if you have one
    
    try:
        # For demonstration, we'll simulate an API response
        if disease.lower() == "fever":
            return "According to simulated medical data: For fever, ensure hydration, rest, and consider over-the-counter fever reducers like paracetamol. Consult a doctor if symptoms worsen or persist or if fever is high/long-lasting."
        elif disease.lower() == "cold":
            return "According to simulated medical data: For common cold, rest, fluids, and symptom relievers (like decongestants). Not a serious condition, but consult a doctor if severe."
        elif disease.lower() == "headache":
            return "According to simulated medical data: For headaches, rest, hydration, and pain relievers. Seek medical attention if severe, sudden, or accompanied by other neurological symptoms."
        else:
            # Simulate API call for unknown diseases
            response = requests.get(API_URL, params={'query': disease}, timeout=5) # Added timeout
            response.raise_for_status() # Raises HTTPError for bad responses (4xx or 5xx)
            data = response.json()
            if data and data.get("guideline"):
                return f"According to external medical data: {data['guideline']}"
            else:
                return f"No specific guidelines found for '{disease}' from external sources. Please consult a medical professional for personalized advice."
    except requests.exceptions.RequestException as e:
        return f"Could not retrieve external medical guidelines due to an API error: {e}. Please try again later or consult a medical professional."
    except Exception as e:
        return f"An unexpected error occurred while processing medical guidelines: {e}"

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
