"""
Igris AI Agent — LangGraph Workflow  (Issue #3 — LangGraph support)

Implements a state-graph workflow for processing user requests:

    ┌──────────┐
    │  START   │
    └────┬─────┘
         │
    ┌────▼─────┐
    │  Parse   │  ← Validate input via Pydantic
    └────┬─────┘
         │
    ┌────▼─────┐
    │  Route   │  ← Detect intent (chat, system_cmd, document, search, math)
    └────┬─────┘
         │
    ┌────▼──────────┐
    │  Execute Tool │  ← Run the appropriate skill tool (if needed)
    └────┬──────────┘
         │
    ┌────▼─────┐
    │ Generate │  ← LLM generates the final response
    └────┬─────┘
         │
    ┌────▼─────┐
    │  Save    │  ← Persist memory atomically
    └────┬─────┘
         │
    ┌────▼─────┐
    │   END    │
    └──────────┘

This graph replaces the flat chain.invoke() call with a structured,
debuggable, and extensible workflow.
"""

from typing import TypedDict, Optional, List, Literal

try:
    from langgraph.graph import StateGraph, END
    LANGGRAPH_AVAILABLE = True
except ImportError:
    LANGGRAPH_AVAILABLE = False

from models.schemas import UserInput


# ──────────────────────────────────────────────
# State definition (TypedDict for LangGraph compatibility)
# ──────────────────────────────────────────────
class WorkflowState(TypedDict, total=False):
    user_input: str
    parsed_text: str
    intent: str
    tool_name: Optional[str]
    tool_result: Optional[str]
    response: str
    error: Optional[str]
    should_save: bool


# ──────────────────────────────────────────────
# Intent keywords for routing
# ──────────────────────────────────────────────
SYSTEM_KEYWORDS = ["shutdown", "reboot", "restart", "sleep", "lock", "lock screen", "hibernate", "turn off"]
SEARCH_KEYWORDS = ["search", "google", "look up", "find online", "what is", "who is", "latest news"]
MATH_KEYWORDS = ["calculate", "compute", "solve", "math", "what is", "how much is"]
DOCUMENT_KEYWORDS = ["read file", "read document", "open file", "summarize file", "load pdf", "read pdf", "read csv"]
UPLOAD_KEYWORDS = ["upload", "cloud", "drive", "google drive", "send to drive"]
FILE_KEYWORDS = ["list files", "list directory", "show files", "write file", "create file", "file info"]


# ──────────────────────────────────────────────
# Node functions
# ──────────────────────────────────────────────
def parse_node(state: WorkflowState) -> WorkflowState:
    """Validate and parse user input using Pydantic."""
    raw = state.get("user_input", "")
    try:
        validated = UserInput(text=raw)
        return {**state, "parsed_text": validated.text, "error": None}
    except Exception as e:
        return {**state, "parsed_text": raw.strip(), "error": f"Input validation note: {e}"}


def route_node(state: WorkflowState) -> WorkflowState:
    """Detect intent from the parsed input text."""
    text = state.get("parsed_text", "").lower()

    if any(kw in text for kw in SYSTEM_KEYWORDS):
        intent = "system_control"
    elif any(kw in text for kw in UPLOAD_KEYWORDS):
        intent = "cloud_upload"
    elif any(kw in text for kw in DOCUMENT_KEYWORDS):
        intent = "document_reader"
    elif any(kw in text for kw in FILE_KEYWORDS):
        intent = "file_manager"
    elif any(kw in text for kw in SEARCH_KEYWORDS):
        intent = "web_search"
    elif any(kw in text for kw in MATH_KEYWORDS):
        # Only if it looks like a math expression or explicit math request
        intent = "math_solver"
    else:
        intent = "chat"

    return {**state, "intent": intent}


def decide_next(state: WorkflowState) -> str:
    """Conditional edge: decide whether we need a tool or go straight to LLM."""
    intent = state.get("intent", "chat")
    if intent == "chat":
        return "generate"
    return "execute_tool"


def execute_tool_node(state: WorkflowState) -> WorkflowState:
    """
    Execute the appropriate skill tool based on intent.
    This is a dispatcher — the actual tool execution happens in the agent's
    tool-calling loop. Here we just tag the state so the main agent knows
    which tools to prefer.
    """
    intent = state.get("intent", "chat")
    text = state.get("parsed_text", "")

    return {
        **state,
        "tool_name": intent,
        "tool_result": f"[{intent}] Tool routing prepared for: {text}",
    }


def generate_node(state: WorkflowState) -> WorkflowState:
    """
    Placeholder for LLM generation.
    In the main agent, this is where chain.invoke() happens.
    The workflow tags the state as ready for generation.
    """
    return {**state, "should_save": True}


def save_node(state: WorkflowState) -> WorkflowState:
    """Mark that memory should be saved after this exchange."""
    return {**state, "should_save": True}


# ──────────────────────────────────────────────
# Build the graph
# ──────────────────────────────────────────────
def build_workflow():
    """Build and compile the LangGraph state graph."""
    if not LANGGRAPH_AVAILABLE:
        return None

    graph = StateGraph(WorkflowState)

    # Add nodes
    graph.add_node("parse", parse_node)
    graph.add_node("route", route_node)
    graph.add_node("execute_tool", execute_tool_node)
    graph.add_node("generate", generate_node)
    graph.add_node("save", save_node)

    # Set entry point
    graph.set_entry_point("parse")

    # Add edges
    graph.add_edge("parse", "route")
    graph.add_conditional_edges("route", decide_next, {"execute_tool": "execute_tool", "generate": "generate"})
    graph.add_edge("execute_tool", "generate")
    graph.add_edge("generate", "save")
    graph.add_edge("save", END)

    return graph.compile()


def process_with_workflow(user_input: str) -> WorkflowState:
    """
    Run user input through the LangGraph workflow.
    Returns the final state with intent, tool routing, etc.

    If LangGraph is not installed, falls back to manual processing.
    """
    initial_state: WorkflowState = {
        "user_input": user_input,
        "parsed_text": "",
        "intent": "chat",
        "tool_name": None,
        "tool_result": None,
        "response": "",
        "error": None,
        "should_save": True,
    }

    workflow = build_workflow()

    if workflow is not None:
        # LangGraph is available — run through the graph
        result = workflow.invoke(initial_state)
        return result
    else:
        # Fallback: run nodes manually in sequence
        state = parse_node(initial_state)
        state = route_node(state)
        if state.get("intent", "chat") != "chat":
            state = execute_tool_node(state)
        state = generate_node(state)
        state = save_node(state)
        return state
