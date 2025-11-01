import os
import streamlit as st
from openai import OpenAI
from langgraph.graph import StateGraph
from typing import TypedDict, Annotated
from graphviz import Digraph

st.set_page_config(page_title="LangGraph Cluster Assistant", page_icon="🤖", layout="wide")


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
# ✅ LangGraph State Definition
# ============================================
class GraphState(TypedDict):
    user_input: str
    result: Annotated[str, "replace"]
    next: str
    category: str

# ============================================
# ✅ Parent Node Decision
# ============================================
def decide_category(state: GraphState):
    text = state["user_input"].lower()
    if any(k in text for k in ["+", "-", "*", "/", "calculate", "sum"]):
        category = "calculator"
    elif any(k in text for k in ["todo", "task", "remind"]):
        category = "manager"
    else:
        category = "ai"
    state["category"] = category
    return {"category": category}

# ============================================
# ✅ CHILD NODE LOGIC
# ============================================

# --- CALCULATOR ---
def calculator_node(state: GraphState):
    expr = state["user_input"].replace("calculate", "").strip()
    try:
        result = eval(expr, {"__builtins__": {}})
        msg = f"🧮 Result: {result}"
    except Exception:
        msg = "⚠️ Could not calculate that."
    return {"result": msg}

# --- MANAGER (To-Do) ---
if "todo_list" not in st.session_state:
    st.session_state.todo_list = []

def todo_node(state: GraphState):
    text = state["user_input"].replace("todo", "").strip()
    if "add" in text:
        task = text.replace("add", "").strip()
        st.session_state.todo_list.append(task)
        msg = f"✅ Added task: {task}"
    elif "show" in text or "list" in text:
        if not st.session_state.todo_list:
            msg = "📝 Your to-do list is empty."
        else:
            msg = "🗂️ To-Do List:\n" + "\n".join([f"- {t}" for t in st.session_state.todo_list])
    else:
        msg = "Use 'todo add <task>' or 'todo show'"
    return {"result": msg}

# --- AI Cluster: Conversation, QA, Summarization, Translation, Sentiment ---
def ai_router(state: GraphState):
    text = state["user_input"].lower()
    if any(k in text for k in ["who is", "what is", "where is", "question"]):
        next_node = "qa"
    elif "translate" in text:
        next_node = "translate"
    elif "summarize" in text or "summary" in text:
        next_node = "summary"
    elif "sentiment" in text:
        next_node = "sentiment"
    else:
        next_node = "conversation"
    return {"next": next_node}

def qa_node(state: GraphState):
    query = state["user_input"]
    try:
        response = client.chat.completions.create(
            model=MODEL_NAME,
            messages=[{"role": "user", "content": query}],
            temperature=0.7,
            max_tokens=200,
        )
        msg = response.choices[0].message.content
    except Exception as e:
        msg = f"❌ QA Error: {str(e)}"
    return {"result": msg}

def conversation_node(state: GraphState):
    query = state["user_input"]
    try:
        response = client.chat.completions.create(
            model=MODEL_NAME,
            messages=[{"role": "user", "content": query}],
            temperature=0.9,
            max_tokens=150,
        )
        msg = response.choices[0].message.content
    except Exception as e:
        msg = f"❌ Conversation error: {str(e)}"
    return {"result": f"💬 {msg}"}

def summary_node(state: GraphState):
    text = state["user_input"]
    try:
        response = client.chat.completions.create(
            model=MODEL_NAME,
            messages=[{"role": "user", "content": f"Summarize this:\n{text}"}],
            temperature=0.5,
            max_tokens=150,
        )
        msg = response.choices[0].message.content
    except Exception as e:
        msg = f"❌ Summary error: {str(e)}"
    return {"result": f"📝 Summary: {msg}"}

def translate_node(state: GraphState):
    text = state["user_input"]
    try:
        response = client.chat.completions.create(
            model=MODEL_NAME,
            messages=[{"role": "user", "content": f"Translate this text to Hindi:\n{text}"}],
            temperature=0.5,
            max_tokens=100,
        )
        msg = response.choices[0].message.content
    except Exception as e:
        msg = f"❌ Translation error: {str(e)}"
    return {"result": f"🌍 Translation: {msg}"}

def sentiment_node(state: GraphState):
    text = state["user_input"]
    try:
        response = client.chat.completions.create(
            model=MODEL_NAME,
            messages=[{"role": "user", "content": f"Analyze sentiment of: {text}"}],
            temperature=0.3,
            max_tokens=100,
        )
        msg = response.choices[0].message.content
    except Exception as e:
        msg = f"❌ Sentiment error: {str(e)}"
    return {"result": f"🧠 Sentiment: {msg}"}

# ============================================
# ✅ BUILD GRAPH
# ============================================
def build_graph():
    graph = StateGraph(GraphState)

    # Main entry decision
    graph.add_node("decide_category", decide_category)

    # Parent Nodes
    graph.add_node("calculator", calculator_node)
    graph.add_node("manager", todo_node)
    graph.add_node("ai_router", ai_router)

    # AI child nodes
    graph.add_node("qa", qa_node)
    graph.add_node("conversation", conversation_node)
    graph.add_node("summary", summary_node)
    graph.add_node("translate", translate_node)
    graph.add_node("sentiment", sentiment_node)

    # Routing logic
    graph.add_conditional_edges(
        "decide_category",
        lambda s: s["category"],
        {
            "calculator": "calculator",
            "manager": "manager",
            "ai": "ai_router",
        },
    )

    graph.add_conditional_edges(
        "ai_router",
        lambda s: s["next"],
        {
            "qa": "qa",
            "conversation": "conversation",
            "summary": "summary",
            "translate": "translate",
            "sentiment": "sentiment",
        },
    )

    graph.set_entry_point("decide_category")
    return graph.compile()

# ============================================
# ✅ GRAPH VISUALIZATION
# ============================================
def draw_graph(active_path=None):
    dot = Digraph()
    dot.attr("node", shape="box", style="rounded,filled", fontname="Arial", fontsize="10")

    def style_node(name, label):
        """Style each node based on active path."""
        if active_path and name in active_path:
            dot.node(name, label, fillcolor="#90EE90", color="#228B22", fontcolor="black")
        else:
            dot.node(name, label, fillcolor="#F8F9FA", color="#555555", fontcolor="black")

    # --- Decision Node ---
    style_node("decide_category", "🧠 Decide Category")

    # --- Calculator Cluster ---
    with dot.subgraph(name="cluster_calculator") as c:
        if active_path and "calculator" in active_path:
            c.attr(label="🧮 Calculator Cluster", color="#228B22", penwidth="3")  # highlight
            c.node("calculator", "🧮 Calculator", fillcolor="#90EE90", color="#228B22")
        else:
            c.attr(label="🧮 Calculator Cluster", color="lightblue", penwidth="1")
            c.node("calculator", "🧮 Calculator", fillcolor="#F8F9FA", color="#555555")

    # --- Manager Cluster ---
    with dot.subgraph(name="cluster_manager") as c:
        if active_path and "manager" in active_path:
            c.attr(label="📋 Manager Cluster", color="#228B22", penwidth="3")
            c.node("manager", "🗂️ To-Do Manager", fillcolor="#90EE90", color="#228B22")
        else:
            c.attr(label="📋 Manager Cluster", color="lightgreen", penwidth="1")
            c.node("manager", "🗂️ To-Do Manager", fillcolor="#F8F9FA", color="#555555")

    # --- AI Cluster ---
    ai_active = any(k in active_path for k in ["ai_router", "qa", "conversation", "summary", "translate", "sentiment"])
    with dot.subgraph(name="cluster_ai") as c:
        if ai_active:
            c.attr(label="🤖 AI Cluster", color="#FF8C00", penwidth="3")  # highlighted orange border
        else:
            c.attr(label="🤖 AI Cluster", color="orange", penwidth="1")

        ai_nodes = {
            "ai_router": "AI Router",
            "qa": "❓ Q&A",
            "conversation": "💬 Conversation",
            "summary": "✂️ Summarizer",
            "translate": "🌍 Translator",
            "sentiment": "🧠 Sentiment",
        }

        for k, v in ai_nodes.items():
            if active_path and k in active_path:
                c.node(k, v, fillcolor="#90EE90", color="#228B22")
            else:
                c.node(k, v, fillcolor="#F8F9FA", color="#555555")

    # --- Edges ---
    edges = [
        ("decide_category", "calculator"),
        ("decide_category", "manager"),
        ("decide_category", "ai_router"),
        ("ai_router", "qa"),
        ("ai_router", "conversation"),
        ("ai_router", "summary"),
        ("ai_router", "translate"),
        ("ai_router", "sentiment"),
    ]

    # Highlight edges in active path
    for src, dst in edges:
        if active_path and src in active_path and dst in active_path:
            dot.edge(src, dst, color="green", penwidth="2")
        else:
            dot.edge(src, dst, color="#999999")

    return dot




# ============================================
# ✅ STREAMLIT UI
# ============================================
st.title("🤖 LangGraph Clustered Assistant")

if "graph" not in st.session_state:
    st.session_state.graph = build_graph()
if "history" not in st.session_state:
    st.session_state.history = []
if "active_path" not in st.session_state:
    st.session_state.active_path = []

query = st.text_input("💬 Enter your query (e.g., 'calculate 25*4', 'todo add buy milk', 'summarize text')")

if st.button("▶ Run Workflow"):
    if not query.strip():
        st.warning("Please enter something.")
    else:
        state = {"user_input": query, "result": "", "next": "", "category": ""}
        path = ["decide_category"]
        cat = decide_category(state)["category"]
        path.append(cat)

        if cat == "calculator":
            result = calculator_node(state)
        elif cat == "manager":
            result = todo_node(state)
        elif cat == "ai":
            next_ai = ai_router(state)["next"]
            path.append(next_ai)
            if next_ai == "qa":
                result = qa_node(state)
            elif next_ai == "summary":
                result = summary_node(state)
            elif next_ai == "translate":
                result = translate_node(state)
            elif next_ai == "sentiment":
                result = sentiment_node(state)
            else:
                result = conversation_node(state)
        else:
            result = {"result": "⚠️ Unknown request."}

        st.session_state.active_path = path
        st.session_state.history.append(result["result"])
        st.success(result["result"])

# History
if st.session_state.history:
    st.subheader("📜 Conversation History")
    for i, h in enumerate(st.session_state.history, 1):
        st.markdown(f"**{i}.** {h}")

# Graph visualization
st.subheader("🧩 LangGraph Flow Visualization")
st.graphviz_chart(draw_graph(st.session_state.active_path))
