# pip install langgrap

from typing import List, TypedDict
from langgraph.graph import StateGraph, END


# ---- Define state schema properly ----
class ConversationState(TypedDict):
    messages: List[str]


# ---- Mock LLM (no API needed) ----
def mock_llm(user_message: str) -> str:
    """Simple rule-based reply instead of real LLM"""
    if "hello" in user_message.lower():
        return "Hi there!  How can I help you?"
    elif "bye" in user_message.lower():
        return "Goodbye!  Have a great day!"
    else:
        return f"I received your message: '{user_message}'"


# ---- Workflow Nodes ----
def process_input(state: ConversationState) -> ConversationState:
    user_message = state["messages"][-1]  # last user message
    print(f"[User Input Node] Received: {user_message}")
    return state


def generate_response(state: ConversationState) -> ConversationState:
    user_message = state["messages"][-1]
    ai_message = mock_llm(user_message)  # use mock LLM
    state["messages"].append(ai_message)
    print(f"[AI Response Node] Generated: {ai_message}")
    return state


# ---- Build Workflow ----
workflow = StateGraph(ConversationState)

workflow.add_node("process_input", process_input)
workflow.add_node("generate_response", generate_response)

# Define edges
workflow.set_entry_point("process_input")
workflow.add_edge("process_input", "generate_response")
workflow.add_edge("generate_response", END)

# Compile
app = workflow.compile()


# ---- Run Example ----
if __name__ == "__main__":
    # Start with a user message inside dict
    state = {"messages": ["hello"]}
    result = app.invoke(state)

    print("\nFinal State:", result)
