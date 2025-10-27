import streamlit as st
from langgraph.graph import StateGraph, END
from typing import TypedDict, Annotated

# --------------------------
# Define the shared graph state
# --------------------------
class GraphState(TypedDict):
    user_input: str
    result: Annotated[str, "replace"]   # replaces value safely each step

# --------------------------
# Node 1 – Decide what to do
# --------------------------
def decide_action(state: GraphState):
    text = state["user_input"].lower()

    if any(op in text for op in ["+", "-", "*", "/", "calculate", "sum", "add", "multiply"]):
        return {"next": "calculator"}
    elif any(q in text for q in ["who is", "what is", "where is"]):
        return {"next": "fact_lookup"}
    else:
        return {"next": "echo_reply"}

# --------------------------
# Node 2 – Calculator
# --------------------------
def calculator_node(state: GraphState):
    expr = state["user_input"].replace("calculate", "").strip()
    try:
        result = eval(expr, {"__builtins__": {}})
        msg = f"🧮 Result: {result}"
    except Exception:
        msg = "⚠️ Sorry, I couldn’t calculate that."
    return {"result": msg}

# --------------------------
# Node 3 – Simple facts
# --------------------------
def fact_lookup_node(state: GraphState):
    text = state["user_input"].lower().strip()
    facts = {
        "who is elon musk": "Elon Musk is the CEO of Tesla and SpaceX.",
        "what is langgraph": "LangGraph lets you build stateful AI workflows.",
        "where is india": "India is a country in South Asia."
    }
    msg = facts.get(text, "I don’t have that fact right now.")
    return {"result": msg}

# --------------------------
# Node 4 – Fallback echo
# --------------------------
def echo_node(state: GraphState):
    msg = f"🗣️ You said: '{state['user_input']}'"
    return {"result": msg}

# --------------------------
# Build the LangGraph
# --------------------------
def build_graph():
    graph = StateGraph(GraphState)

    graph.add_node("decide", decide_action)
    graph.add_node("calculator", calculator_node)
    graph.add_node("fact_lookup", fact_lookup_node)
    graph.add_node("echo_reply", echo_node)

    # connect logic
    graph.add_conditional_edges(
        "decide",
        lambda s: s["next"],
        {
            "calculator": "calculator",
            "fact_lookup": "fact_lookup",
            "echo_reply": "echo_reply"
        }
    )
    graph.set_entry_point("decide")
    return graph.compile()

# --------------------------
# Streamlit UI
# --------------------------
st.set_page_config(page_title="LangGraph Smart Assistant", page_icon="🧠")
st.title("🧠 LangGraph Smart Assistant (Offline Demo)")

if "graph" not in st.session_state:
    st.session_state.graph = build_graph()
if "history" not in st.session_state:
    st.session_state.history = []

query = st.text_input("Ask me something (e.g., 'calculate 25*4' or 'Who is Elon Musk?')")

if st.button("Run"):
    if not query.strip():
        st.warning("Please type something first.")
    else:
        result = st.session_state.graph.invoke({"user_input": query})
        st.session_state.history.append(result["result"])
        st.success(result["result"])

if st.session_state.history:
    st.subheader("🗂 Conversation History")
    for i, h in enumerate(st.session_state.history, 1):
        st.text(f"{i}. {h}")
