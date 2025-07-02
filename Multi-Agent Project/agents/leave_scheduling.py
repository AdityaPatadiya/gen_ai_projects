import os
import json
from datetime import date
import sqlite3
from langchain_core.messages import HumanMessage, AIMessage, SystemMessage
from langchain_openai import AzureChatOpenAI
from langchain.prompts import ChatPromptTemplate
from langchain_core.tools import tool

from core.state import AgentState
from utils.db_setup import DATABASE_FILE

# Initialize LLM
llm = AzureChatOpenAI(
    azure_endpoint=os.getenv("AZURE_OPENAI_API_BASE"),
    api_key=os.getenv("AZURE_API_KEY"),
    api_version=os.getenv("AZURE_OPENAI_API_VERSION"),
    deployment_name=os.getenv("AZURE_OPENAI_API_NAME"),
    model_name="gpt-4o",
    max_tokens=300,
)

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
    conn = sqlite3.connect(DATABASE_FILE)
    cursor = conn.cursor()
    cursor.execute("SELECT annual_leave_balance, sick_leave_balance FROM employees WHERE employee_id = ?", (employee_id,))
    result = cursor.fetchone()
    conn.close()

    if result:
        annual, sick = result
        return f"Employee {employee_id} has {annual} days of annual leave and {sick} days of sick leave remaining."
    return f"Employee ID '{employee_id}' not found in the system."

@tool
def submit_leave_request(employee_id: str, start_date: str, end_date: str, reason: str) -> str:
    """
    Submits a leave request for an employee. Dates should be in YYYY-MM-DD format.
    Args:
        employee_id (str): The ID of the employee.
        start_date (str): The start date of the leave (YYYY-MM-DD).
        end_date (str): The end date of the leave (YYYY-MM-DD).
        reason (str): The reason for the leave (e.g., "vacation", "sick leave").
    Returns:
        str: A confirmation or error message.
    """
    print(f"---TOOL CALL: submit_leave_request for EMP: {employee_id}, Dates: {start_date}-{end_date}, Reason: {reason}---")
    try:
        # Basic date validation
        s_date = date.fromisoformat(start_date)
        e_date = date.fromisoformat(end_date)
        if s_date > e_date:
            return "Leave request failed: Start date cannot be after end date."
        
        conn = sqlite3.connect(DATABASE_FILE)
        cursor = conn.cursor()
        
        # First, check if employee exists
        cursor.execute("SELECT employee_id FROM employees WHERE employee_id = ?", (employee_id,))
        if not cursor.fetchone():
            conn.close()
            return f"Leave request failed: Employee ID '{employee_id}' not found."

        cursor.execute(
            "INSERT INTO leave_requests (employee_id, start_date, end_date, reason, status) VALUES (?, ?, ?, ?, ?)",
            (employee_id, start_date, end_date, reason, 'Pending')
        )
        conn.commit()
        conn.close()
        return f"Leave request for Employee {employee_id} from {start_date} to {end_date} for '{reason}' submitted successfully and is pending approval."
    except ValueError:
        return "Leave request failed: Invalid date format. Please use YYYY-MM-DD."
    except Exception as e:
        return f"Leave request failed due to an internal database error: {e}"

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
        "Use current date and time." # Provide context for date understanding
    )

    # In a real system, you'd use LLM's function calling capabilities here.
    # For now, we'll keep the direct keyword-based triggering as in your original code,
    # but the tool functions themselves are now backed by a database.
    response_content = ""

    # Use LLM with tools bound to it for intelligent tool use
    # For this to work, the LLM needs to be able to "decide" to call tools
    # This requires using the LLM in a way that enables tool calling.
    # Langchain's `create_react_agent` or binding tools directly to the LLM and processing `tool_calls`
    # in the output is the standard way. Let's simplify for now, keeping your structure,
    # but remember this is the area for advanced NLU.

    # Example of how you would set up LLM with tools
    llm_with_tools = llm.bind_tools([check_leave_balance, submit_leave_request])

    # The prompt should encourage the LLM to use tools
    prompt = ChatPromptTemplate.from_messages([
        SystemMessage(content=system_prompt),
        HumanMessage(content=last_human_message)
    ])

    # Invoke the LLM with the bound tools
    llm_response = llm_with_tools.invoke(prompt.format_messages())

    # Check if the LLM decided to call a tool
    tool_calls = llm_response.additional_kwargs.get("tool_calls", [])
    if tool_calls:
        for tool_call in tool_calls:
            print("Tool call received:", tool_call)
            tool_name = tool_call.get('name')
            raw_args = tool_call.get('args', {})
            tool_args = json.loads(raw_args) if isinstance(raw_args, str) else raw_args

            if tool_name == "check_leave_balance":
                response_content = check_leave_balance(tool_args.get("employee_id"))
            elif tool_name == "submit_leave_request":
                # Unpack arguments for submit_leave_request
                response_content = submit_leave_request(
                    tool_args.get("employee_id"),
                    tool_args.get("start_date"),
                    tool_args.get("end_date"),
                    tool_args.get("reason")
                )
            else:
                response_content = f"Tool '{tool_name}' not recognized."
    else:
        response_content = llm_response.content

    return {"messages": [AIMessage(content=response_content)]}
