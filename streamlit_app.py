import os
import streamlit as st
from openai import OpenAI
from langgraph.graph import StateGraph
from typing import TypedDict, Annotated
from graphviz import Digraph



HF_TOKEN = ""

if not HF_TOKEN:
    st.sidebar.title("🔑 API Configuration")
    HF_TOKEN = st.sidebar.text_input("Enter your Hugging Face Token:", type="password")
    if HF_TOKEN:
        os.environ["HF_TOKEN"] = HF_TOKEN

client = OpenAI(
    base_url="https://router.huggingface.co/v1",
    api_key=HF_TOKEN,
)

MODEL_NAME = "katanemo/Arch-Router-1.5B:hf-inference"

# ============================================
# ✅ LANGGRAPH LOGIC
# ============================================

class GraphState(TypedDict):
    user_input: str
    result: Annotated[str, "replace"]
    next: str

# --------------------------
# Node Functions
# --------------------------
def decide_action(state: GraphState):
    text = state["user_input"].lower()
    if any(op in text for op in ["+", "-", "*", "/", "calculate", "sum", "add", "multiply"]):
        state["next"] = "calculator"
    elif any(q in text for q in ["who is", "what is", "where is"]):
        state["next"] = "fact_lookup"
    else:
        state["next"] = "echo_reply"
    return {"next": state["next"]}

def calculator_node(state: GraphState):
    expr = state["user_input"].replace("calculate", "").strip()
    try:
        result = eval(expr, {"__builtins__": {}})
        msg = f"🧮 Result: {result}"
    except Exception:
        msg = "⚠️ Could not calculate that."
    return {"result": msg}

# ✅ REPLACED fact_lookup WITH HF CHATBOT CALL
def fact_lookup_node(state: GraphState):
    query = state["user_input"]

    try:
        # Prepare message history for contextual replies
        messages = [{"role": "user", "content": query}]

        response = client.chat.completions.create(
            model=MODEL_NAME,
            messages=messages,
            temperature=0.7,
            max_tokens=300,
        )

        msg = response.choices[0].message.content
    except Exception as e:
        msg = f"❌ Chatbot error: {str(e)}"

    return {"result": msg}

def echo_node(state: GraphState):
    msg = f"🗣️ You said: '{state['user_input']}'"
    return {"result": msg}

# --------------------------
# Build Graph
# --------------------------
def build_graph():
    graph = StateGraph(GraphState)
    graph.add_node("decide", decide_action)
    graph.add_node("calculator", calculator_node)
    graph.add_node("fact_lookup", fact_lookup_node)
    graph.add_node("echo_reply", echo_node)

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
# Visualization Helper
# --------------------------
def draw_graph(active_path=None):
    dot = Digraph()
    dot.attr("node", shape="box", style="rounded,filled", fontname="Arial")

    nodes = {
        "decide": "🧠 Decide Action",
        "calculator": "🧮 Calculator",
        "fact_lookup": "💬 Hugging Face QA",
        "echo_reply": "💬 Echo Reply"
    }

    for key, label in nodes.items():
        if active_path and key in active_path:
            dot.node(key, label, fillcolor="#90EE90", color="#228B22", fontcolor="black")
        else:
            dot.node(key, label, fillcolor="#F8F9FA", color="#555555", fontcolor="black")

    dot.edge("decide", "calculator", label="if math keywords")
    dot.edge("decide", "fact_lookup", label="if who/what/where")
    dot.edge("decide", "echo_reply", label="otherwise")
    return dot

# ============================================
# ✅ STREAMLIT UI
# ============================================
st.set_page_config(page_title="LangGraph + HF QA", page_icon="🧠", layout="wide")
st.title("🧠 LangGraph Smart Assistant + 🤖 Hugging Face Chatbot")

if "graph" not in st.session_state:
    st.session_state.graph = build_graph()
if "history" not in st.session_state:
    st.session_state.history = []
if "active_path" not in st.session_state:
    st.session_state.active_path = []

query = st.text_input("💬 Enter your query (e.g., 'calculate 25*4', 'Who is Elon Musk')")

if st.button("▶ Run Workflow"):
    if not query.strip():
        st.warning("Please enter something.")
    else:
        state = {"user_input": query, "result": "", "next": ""}
        path = ["decide"]
        next_step = decide_action(state)["next"]

        if next_step == "calculator":
            result = calculator_node(state)
        elif next_step == "fact_lookup":
            result = fact_lookup_node(state)
        else:
            result = echo_node(state)

        path.append(next_step)
        st.session_state.active_path = path
        st.session_state.history.append(result["result"])
        st.success(result["result"])

# Conversation History
if st.session_state.history:
    st.subheader("🗂 Conversation History")
    for i, h in enumerate(st.session_state.history, 1):
        st.markdown(f"**{i}.** {h}")

# Flow Visualization
st.subheader("🧩 LangGraph Flow Visualization")
st.graphviz_chart(draw_graph(st.session_state.active_path))
