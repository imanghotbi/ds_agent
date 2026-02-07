from langgraph.graph import StateGraph, START, END

from ds_agent.core.state import AgentState
from ds_agent.config import Nodes
from ds_agent.core.nodes.supervisor import supervisor_node
from ds_agent.core.nodes.worker import cleaner_node, eda_node, feature_engineer_node, trainer_node, storyteller_node
from ds_agent.core.nodes.tools import tool_node
from ds_agent.core.nodes.reporter import reporter_node

# --- Conditional Logic ---

def router(state: AgentState) -> str:
    return state["next"]

def worker_router(state: AgentState) -> str:
    last_message = state['messages'][-1]
    if hasattr(last_message, 'tool_calls') and last_message.tool_calls:
        return Nodes.TOOLS
    return Nodes.SUPERVISOR

def tool_router(state: AgentState) -> str:
    # Returns the agent that initiated the tool call
    return state.get("next")

# --- Graph ---

def create_graph() -> StateGraph:
    workflow = StateGraph(AgentState)
    
    workflow.add_node(Nodes.SUPERVISOR, supervisor_node)
    workflow.add_node(Nodes.CLEANER, cleaner_node)
    workflow.add_node(Nodes.EDA, eda_node)
    workflow.add_node(Nodes.FEATURE_ENGINEER, feature_engineer_node)
    workflow.add_node(Nodes.TRAINER, trainer_node)
    workflow.add_node(Nodes.STORYTELLER, storyteller_node)
    workflow.add_node(Nodes.TOOLS, tool_node)
    workflow.add_node(Nodes.REPORTER, reporter_node)
    
    workflow.add_edge(START, Nodes.SUPERVISOR)
    
    workflow.add_conditional_edges(
        Nodes.SUPERVISOR,
        router,
        {
            Nodes.CLEANER: Nodes.CLEANER,
            Nodes.EDA: Nodes.EDA,
            Nodes.FEATURE_ENGINEER: Nodes.FEATURE_ENGINEER,
            Nodes.TRAINER: Nodes.TRAINER,
            Nodes.STORYTELLER: Nodes.STORYTELLER,
            Nodes.REPORTER: Nodes.REPORTER,
            Nodes.FINISH: Nodes.REPORTER  # Route FINISH to Reporter
        }
    )
    
    workflow.add_conditional_edges(
        Nodes.CLEANER,
        worker_router,
        {Nodes.TOOLS: Nodes.TOOLS, Nodes.SUPERVISOR: Nodes.SUPERVISOR}
    )
    
    workflow.add_conditional_edges(
        Nodes.EDA,
        worker_router,
        {Nodes.TOOLS: Nodes.TOOLS, Nodes.SUPERVISOR: Nodes.SUPERVISOR}
    )

    workflow.add_conditional_edges(
        Nodes.FEATURE_ENGINEER,
        worker_router,
        {Nodes.TOOLS: Nodes.TOOLS, Nodes.SUPERVISOR: Nodes.SUPERVISOR}
    )

    workflow.add_conditional_edges(
        Nodes.TRAINER,
        worker_router,
        {Nodes.TOOLS: Nodes.TOOLS, Nodes.SUPERVISOR: Nodes.SUPERVISOR}
    )

    workflow.add_conditional_edges(
        Nodes.STORYTELLER,
        worker_router,
        {Nodes.TOOLS: Nodes.TOOLS, Nodes.SUPERVISOR: Nodes.SUPERVISOR}
    )
    
    workflow.add_conditional_edges(
        Nodes.TOOLS,
        tool_router,
        {
            Nodes.CLEANER: Nodes.CLEANER, 
            Nodes.EDA: Nodes.EDA,
            Nodes.FEATURE_ENGINEER: Nodes.FEATURE_ENGINEER,
            Nodes.TRAINER: Nodes.TRAINER,
            Nodes.STORYTELLER: Nodes.STORYTELLER
        }
    )
    
    workflow.add_edge(Nodes.REPORTER, END)
    
    return workflow.compile()