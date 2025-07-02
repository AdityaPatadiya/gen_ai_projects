import os
from dotenv import load_dotenv
from langchain_core.messages import HumanMessage

from langgraph.graph import StateGraph, END, START

# Import from your custom core and agents packages
from core.state import AgentState
from core.router import route_agent
from agents.clinical_assistant import clinical_assistant_agent
from agents.leave_scheduling import leave_scheduling_agent

# Load environment variables from .env file
load_dotenv()

# --- Fallback Agent (for unhandled queries) ---
def fallback_agent(state: AgentState):
    """
    Handles queries that are not routed to specific agents.
    """
    print("---FALLBACK AGENT ACTIVATED---")
    return {"messages": [HumanMessage(content="I'm sorry, I couldn't understand your request. Please ask about health-related queries or leave management.")]}

# --- Build the LangGraph Workflow ---
def build_graph():
    builder = StateGraph(AgentState)

    # Add nodes for each agent
    builder.add_node("router", route_agent) # The router is also a node
    builder.add_node("clinical_assistant", clinical_assistant_agent)
    builder.add_node("leave_scheduling", leave_scheduling_agent)
    builder.add_node("fallback", fallback_agent)

    # Set the entry point of the graph
    builder.set_entry_point("router") # All initial inputs go to the router

    # Define conditional edges from the router
    builder.add_conditional_edges(
        "router",
        # This lambda function takes the output of the 'router' node
        # (which is the dict name of the next agent) and uses it
        # to determine the next node.
        lambda state: state["__route__"],
        {
            "clinical_assistant": "clinical_assistant",
            "leave_scheduling": "leave_scheduling",
            "fallback": "fallback",
        }
    )

    # After an agent processes, for simplicity, we'll end the graph.
    # In a more complex system, they might go back to the router for follow-up,
    # or to a different node for validation/confirmation.
    builder.add_edge("clinical_assistant", END)
    builder.add_edge("leave_scheduling", END)
    builder.add_edge("fallback", END)

    # Compile the graph
    graph = builder.compile()
    return graph

# --- Main Execution ---
if __name__ == "__main__":
    # Ensure OPENAI_API_KEY is set
    if not os.getenv("OPENAI_API_KEY"):
        raise ValueError("OPENAI_API_KEY environment variable not set. Please set it in your .env file or system environment.")

    ai_graph = build_graph()

    print("--- Multi-Agent Bot Started ---")
    print("Type 'exit' or 'quit' to end the conversation.")

    while True:
        user_input = input("\nYou: ")
        if user_input.lower() in ["exit", "quit"]:
            print("--- Bot: Goodbye! ---")
            break

        # Invoke the graph with the user's message
        inputs = {"messages": [HumanMessage(content=user_input)]}

        try:
            # We use .stream() to see the progression, but .invoke() also works.
            # For a production chatbot, you might only want the final result.
            final_output = None
            for s in ai_graph.stream(inputs):
                if "__end__" not in s:
                    # Print intermediate steps for debugging/demonstration
                    # print(s) 
                    pass
                else:
                    final_output = s["__end__"]
            
            if final_output and final_output['messages']:
                print(f"Bot: {final_output['messages'][-1].content}")
            else:
                print("Bot: An unexpected error occurred or no response was generated.")

        except Exception as e:
            print(f"Bot: An error occurred: {e}")
            print("Please try again.")
