from langchain_core.messages import SystemMessage, ToolMessage
from langgraph.graph import StateGraph, START
from langgraph.prebuilt import tools_condition

from e2b_code_interpreter import AsyncSandbox

from ds_agent.core.state import AgentState
from ds_agent.tools.e2b import E2BTools
from ds_agent.config import settings
from ds_agent.utils.logger import logger
from ds_agent.core.prompts import DATA_CLEANING_PROMPT
from ds_agent.core.llm import LLMFactory

# --- Nodes ---

async def agent_node(state: AgentState):
    """
    The main agent node that calls the LLM.
    """
    logger.info("Agent node started")
    llm_factory = LLMFactory()
    llm = llm_factory.create()
    
    # Static tool definitions for binding
    tool_defs = E2BTools(None).get_tools()
    llm_with_tools = llm.bind_tools(tool_defs)
    
    # Apply retry logic if configured
    if llm_factory.max_retries > 0:
        llm_with_tools = llm_with_tools.with_retry(stop_after_attempt=llm_factory.max_retries)
    
    messages = state['messages']
    if not messages or not isinstance(messages[0], SystemMessage):
        messages = [SystemMessage(content=DATA_CLEANING_PROMPT)] + messages
        
    response = await llm_with_tools.ainvoke(messages)
    return {"messages": [response]}

async def tool_node(state: AgentState):
    """
    Executes the tools requested by the LLM.
    Manages the E2B AsyncSandbox session.
    """
    logger.info("Tool node started")
    sandbox = state.get('sandbox_session')
    if not sandbox:
        logger.info("Initializing E2B AsyncSandbox...")
        sandbox = await AsyncSandbox.create(api_key=settings.e2b_api_key)
        
    new_cells = []
    def update_callback(cell_data):
        new_cells.append(cell_data)

    # Initialize Tools with the active sandbox
    e2b_tools = E2BTools(sandbox, update_state_callback=update_callback)
    tool_map = {t.name: t for t in e2b_tools.get_tools()}
    
    last_message = state['messages'][-1]
    results = []
    
    for tool_call in last_message.tool_calls:
        tool_name = tool_call['name']
        tool_args = tool_call['args']
        tool_id = tool_call['id']
        
        logger.info(f"Executing tool: {tool_name}")
        if tool_name in tool_map:
            tool_instance = tool_map[tool_name]
            try:
                output = await tool_instance.ainvoke(tool_args)
            except Exception as e:
                logger.error(f"Error executing {tool_name}: {e}")
                output = f"Error executing tool: {e}"
                
            results.append(ToolMessage(tool_call_id=tool_id, name=tool_name, content=str(output)))
        else:
            logger.warning(f"Tool not found: {tool_name}")
            results.append(ToolMessage(tool_call_id=tool_id, name=tool_name, content="Error: Tool not found"))

    return {
        "messages": results,
        "notebook_cells": new_cells,
        "sandbox_session": sandbox
    }

# --- Graph Definition ---

def create_graph():
    workflow = StateGraph(AgentState)
    
    workflow.add_node("agent", agent_node)
    workflow.add_node("tools", tool_node)
    
    workflow.add_edge(START, "agent")
    
    workflow.add_conditional_edges(
        "agent",
        tools_condition,
    )
    
    workflow.add_edge("tools", "agent")
    
    return workflow.compile()
