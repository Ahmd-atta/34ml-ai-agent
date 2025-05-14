"""
Universal LangGraph builder for 34ML AI Agent
With improved recursion control and simplified graph structure
"""

from typing import TypedDict, Any, Callable, Dict, List, Optional
import inspect
import logging

from langgraph.graph import StateGraph, END
from agents.orchestrator import orchestrator
from agents.graph_nodes import generator_node, scheduler_node, kb_node

# Set up logging
logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)

# ------------------------------------------------------------------
# Shared state schema
# ------------------------------------------------------------------
class GraphState(TypedDict, total=False):
    user_input: str
    route: str
    result: Any  # Can be string or dict (e.g., {"draft": "...", "image_url": "..."})
    generated: bool
    conversation_history: List[Dict[str, str]]  # [{"user": "...", "bot": "..."}]
    approved_post: Optional[str]  # For HITL-approved content
    image_url: Optional[str]  # For images
    image_path: Optional[str]
    waiting_for_qa: bool  # Flag to indicate QA/HITL is pending
    draft: Optional[str]  # Store the draft during QA/HITL
    qa_processed: bool  # Flag to indicate QA/HITL processing completed
       # NEW ──────────────────────────────
    channel: str        # "Instagram", "LinkedIn", …
    image_done: bool    # True after first DALL·E call


# ------------------------------------------------------------------
# Helper: add_conditional_edges adapter
# ------------------------------------------------------------------
def _add_branching(
    g: StateGraph,
    node_name: str,
    selector: Callable[[Dict[str, Any]], str],
    edge_map: Dict[str, str],
):
    """
    Call add_conditional_edges() using whichever signature
    the local langgraph version supports.
    """
    sig = inspect.signature(g.add_conditional_edges)
    params = list(sig.parameters)

    if "condition" in params:  # >=0.4.4  (not on PyPI yet)
        g.add_conditional_edges(node_name, condition=selector, edge_map=edge_map)
    elif len(params) == 3:
        # <=0.4.2   (node_name, edge_map, selector_fn)
        g.add_conditional_edges(node_name, edge_map, selector)
    else:
        # <=0.4.0   (node_name, selector_fn, edge_map)
        g.add_conditional_edges(node_name, selector, edge_map)


# ------------------------------------------------------------------
# QA/HITL Node
# ------------------------------------------------------------------
def qa_hitl_node(state: GraphState) -> GraphState:
    """
    Handle QA/HITL commands (approve, edit, reject, quit) based on user input.
    """
    logger.debug(f"qa_hitl_node state keys: {list(state.keys())}")
    
    # Mark as processed to avoid recursion
    state["qa_processed"] = True
    
    if not state.get("waiting_for_qa"):
        # Not in QA mode, just pass through
        logger.debug("Not waiting for QA, passing through")
        return state

    user_input = state["user_input"].lower().strip()
    draft = state.get("draft", "")

    if user_input in ["approve", "a"]:
        state["approved_post"] = draft
        state["result"] = "✅ Saved & approved"
        state["waiting_for_qa"] = False
    elif user_input in ["reject", "r"]:
        state["result"] = "Draft rejected."
        state["waiting_for_qa"] = False
    elif user_input in ["quit", "q"]:
        state["result"] = "Action cancelled."
        state["waiting_for_qa"] = False
    elif user_input.startswith("edit ") or user_input.startswith("e "):
        new_text = user_input[5:] if user_input.startswith("edit ") else user_input[2:]
        new_text = new_text.strip()
        if new_text:
            state["approved_post"] = new_text
            state["result"] = f"Draft updated: {new_text}"
            state["waiting_for_qa"] = False
        else:
            state["result"] = "Please provide text to edit the draft."
    else:
        state["result"] = "Invalid command. Please use: approve (a), edit <new text> (e <new text>), reject (r), or quit (q)."

    # Clear QA/HITL state if done
    if not state.get("waiting_for_qa", True):
        state["draft"] = None
        state["image_url"] = None
    
    return state


# ------------------------------------------------------------------
# Simplified orchestrator to end node selector
# ------------------------------------------------------------------
def orchestrator_selector(state: Dict[str, Any]) -> str:
    """
    Determine the next node from orchestrator.
    Forces a simple path to prevent infinite loops.
    """
    logger.debug(f"orchestrator_selector state keys: {list(state.keys())}")
    
    route = state.get("route", "")
    
    if route == "generate":
        return "generator"
    elif route == "scheduler":
        return "scheduler"
    elif route == "kb":
        return "kb"
    else:
        # Default end - important to have a fallback
        logger.debug("No matching route, going to END")
        return "end"


# ------------------------------------------------------------------
# Simplified leaf node selector that always goes to END
# ------------------------------------------------------------------
def always_end_selector(state: Dict[str, Any]) -> str:
    """
    Always go to END from leaf nodes - simplifies graph flow
    to prevent recursion issues.
    """
    logger.debug("Leaf node selector: going to END")
    return "end"


# ------------------------------------------------------------------
# Special selector for generator that can go to QA/HITL
# ------------------------------------------------------------------
def generator_selector(state: Dict[str, Any]) -> str:
    """
    Determine whether to go to QA/HITL or END after generation.
    """
    logger.debug(f"generator_selector state keys: {list(state.keys())}")
    
    if state.get("waiting_for_qa") and not state.get("qa_processed"):
        logger.debug("Generator produced draft, going to QA/HITL")
        return "qa_hitl"
    
    logger.debug("Generator done, going to END")
    return "end"


# ------------------------------------------------------------------
# Special selector for QA/HITL based on completion status
# ------------------------------------------------------------------
def qa_hitl_selector(state: Dict[str, Any]) -> str:
    """
    Always go to END from QA/HITL to prevent recursion.
    """
    logger.debug("QA/HITL selector: going to END")
    return "end"


# ------------------------------------------------------------------
# Build & compile
# ------------------------------------------------------------------
def build_graph():
    g = StateGraph(GraphState)

    # nodes --------------------------------------------------------
    g.add_node("orchestrator", orchestrator)
    g.add_node("generator", generator_node)
    g.add_node("scheduler", scheduler_node)
    g.add_node("kb", kb_node)
    g.add_node("qa_hitl", qa_hitl_node)

    # Simplified graph connections - prevent cycles -----------------
    
    # Orchestrator to action nodes
    _add_branching(
        g,
        "orchestrator",
        selector=orchestrator_selector,
        edge_map={
            "generator": "generator",
            "scheduler": "scheduler",
            "kb": "kb",
            "end": END,
        },
    )
    
    # Generator can go to QA/HITL or END
    _add_branching(
        g,
        "generator",
        selector=generator_selector,
        edge_map={
            "qa_hitl": "qa_hitl",
            "end": END,
        },
    )
    
    # Scheduler and KB always go to END
    for leaf_node in ["scheduler", "kb"]:
        _add_branching(
            g,
            leaf_node,
            selector=always_end_selector,
            edge_map={"end": END},
        )
    
    # QA/HITL always goes to END
    _add_branching(
        g,
        "qa_hitl",
        selector=qa_hitl_selector,
        edge_map={"end": END},
    )

    # entry point ------------------------------------------------
    g.set_entry_point("orchestrator")
    return g


# convenience for app.py / tests
def get_runner(checkpointer=None):
    """
    Compile and return the graph runner, optionally with a checkpointer.
    """
    graph = build_graph()
    return graph.compile(checkpointer=checkpointer)