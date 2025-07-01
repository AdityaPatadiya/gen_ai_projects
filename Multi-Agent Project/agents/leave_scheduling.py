import os
from typing import List, Annotated, TypedDict
from datetime import date
from langchain_core.messages import HumanMessage, AIMessage, SystemMessage
from langchain_openai import AzureChatOpenAI
from langchain.prompts import ChatPromptTemplate
from langchain_core.tools import tool

from core.state import AgentState

# Initialize LLM
llm = AzureChatOpenAI(model="gpt-4o", temperature=0.7)

# --- Define Tools for Leave Scheduling Assistant ---
@tool
def check_leave_balance(employee_id: str) -> str:
    """
    Checks the leave balance (e.g., annual, sick) for a given employee ID.
    Args:
        employee_id (str): The ID of the employee.
    Returns:
        str: A message indicating the leave balance.
    """
    print(f"---TOOL CALL: check_leave_balance for {employee_id}---")
    # This is a dummy implementation. In a real system, it would query an HR database.
    if employee_id == "EMP001":
        return "Employee EMP001 has 15 days of annual leave and 5 days of sick leave remaining. Last updated: 2025-06-30."
    elif employee_id == "EMP002":
        return "Employee EMP002 has 10 days of annual leave and 3 days of sick leave remaining. Last updated: 2025-06-30."
    else:
        return f"Employee ID '{employee_id}' not found or balance unavailable in our dummy system."

@tool
def submit_leave_request(args: dict) -> str:
    """
    Submits a leave request for an employee. Dates should be in YYYY-MM-DD format.
    Args:
        args (dict): A dictionary with keys 'employee_id', 'start_date', 'end_date', 'reason'.
    Returns:
        str: A confirmation or error message.
    """
    employee_id = args.get("employee_id")
    start_date = args.get("start_date")
    end_date = args.get("end_date")
    reason = args.get("reason")
    print(f"---TOOL CALL: submit_leave_request for EMP: {employee_id}, Dates: {start_date}-{end_date}, Reason: {reason}---")
    # This is a dummy implementation. In a real system, it would update an HR database.
    try:
        # Basic date validation
        if not start_date or not end_date:
            return "Leave request failed: Start date and end date are required."
        date.fromisoformat(start_date)
        date.fromisoformat(end_date)
        if employee_id not in ["EMP001", "EMP002"]:
            return "Leave request failed: Invalid Employee ID."
        return f"Leave request for Employee {employee_id} from {start_date} to {end_date} for '{reason}' submitted successfully and is pending approval."
    except ValueError:
        return "Leave request failed: Invalid date format. Please use YYYY-MM-DD."
    except Exception as e:
        return f"Leave request failed due to an internal error: {e}"

# --- Leave Scheduling Agent Node ---
def leave_scheduling_agent(state: AgentState):
    """
    Processes leave-related queries using an LLM and leave management tools.
    """
    print("---LEAVE SCHEDULING AGENT PROCESSING---")
    messages = state['messages']
    last_human_message = messages[-1].content if messages and isinstance(messages[-1], HumanMessage) else ""
    if isinstance(last_human_message, list):
        # Convert list to string for processing
        last_human_message = " ".join(str(item) for item in last_human_message)

    # Define the system prompt for the leave agent
    system_prompt = (
        "You are a polite and efficient leave scheduling assistant. "
        "You can help check leave balances and submit leave requests. "
        "Always ask for necessary details if they are missing (e.g., employee ID, dates, reason)."
        "When submitting a leave request, confirm all details with the user first."
    )

    # For simplicity, we'll use keywords to trigger dummy tool calls.
    # In a production system, you'd use LLM's function calling capabilities.
    response_content = ""
    if "check balance" in last_human_message.lower() and "employee id" in last_human_message.lower():
        # Dummy extraction of employee ID (replace with robust NLU)
        parts = last_human_message.lower().split("employee id")
        employee_id = parts[1].split()[0].upper().strip("?.,!") if len(parts) > 1 else "UNKNOWN"
        if employee_id:
            balance = check_leave_balance(employee_id)
            response_content = f"Here is the leave balance: {balance}"
        else:
            response_content = "Please provide the employee ID to check the balance."
    elif "submit leave" in last_human_message.lower() or "request leave" in last_human_message.lower():
        # Dummy extraction of details (replace with robust NLU/structured output from LLM)
        employee_id = "EMP001"
        start_date = "2025-07-15"
        end_date = "2025-07-20"
        reason = "vacation"

        # Call the underlying function directly to avoid the @tool decorator's signature enforcement
        submission_status = submit_leave_request.invoke({
            "employee_id": employee_id,
            "start_date": start_date,
            "end_date": end_date,
            "reason": reason
        })
        response_content = f"Attempting to submit leave. {submission_status}"
    else:
        # If no specific tool is triggered, just use the LLM for general response
        prompt = ChatPromptTemplate.from_messages([
            SystemMessage(content=system_prompt),
            HumanMessage(content=last_human_message)
        ])
        prompt_value = prompt.format()
        llm_response = llm.invoke(prompt_value)
        response_content = llm_response.content

    return {"messages": [AIMessage(content=response_content)]}
