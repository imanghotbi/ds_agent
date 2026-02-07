from typing import Dict, Any, List
from langchain_core.messages import SystemMessage
from langchain_core.runnables import RunnableConfig
from e2b_code_interpreter import AsyncSandbox

from ds_agent.core.state import AgentState
from ds_agent.tools.e2b import E2BTools
from ds_agent.config import settings
from ds_agent.utils.logger import logger
from ds_agent.core.llm import LLMFactory

def get_llm():
    """
    Creates a configured LLM instance using the LLMFactory.
    """
    llm_factory = LLMFactory()
    llm = llm_factory.create()
    if llm_factory.max_retries > 0:
        return llm.with_retry(stop_after_attempt=llm_factory.max_retries)
    return llm

def get_sandbox(config: RunnableConfig) -> AsyncSandbox:
    """
    Retrieves the sandbox session from the configuration.
    """
    sandbox = config.get("configurable", {}).get("sandbox")
    if not sandbox:
        raise ValueError("Sandbox not found in config. Ensure 'sandbox' is passed in 'configurable'.")
    return sandbox

async def run_worker(state: AgentState, system_prompt: str, sender_name: str) -> Dict[str, Any]:
    """
    Generic worker execution logic.
    
    Args:
        state: The current agent state.
        system_prompt: The persona/instructions for this worker.
        sender_name: The name of the worker (used for tracking).
        
    Returns:
        Dict update for the state.
    """
    logger.info(f"{sender_name} agent started")
    llm = get_llm()
    
    # We instantiate tools with None just to get definitions for binding
    tool_defs = E2BTools(None).get_tools()
    llm_with_tools = llm.bind_tools(tool_defs)
    
    # Inject Supervisor Instructions if available
    instructions = state.get("supervisor_instructions", "")
    if instructions:
        system_prompt = f"{system_prompt}\n\n### MANAGER INSTRUCTIONS ###\n{instructions}"
    
    # Prepend the specialized system prompt to the message history
    current_messages = [SystemMessage(content=system_prompt)] + state['messages']
    
    response = await llm_with_tools.ainvoke(current_messages)
    return {"messages": [response], "sender": sender_name}
