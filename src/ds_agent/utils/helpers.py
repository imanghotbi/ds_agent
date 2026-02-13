from typing import Dict, Any, List, Optional, Type, Union, Tuple
from pydantic import BaseModel, ValidationError
from langchain_core.messages import SystemMessage, BaseMessage
from langchain_core.runnables import RunnableConfig
from e2b_code_interpreter import AsyncSandbox

from ds_agent.core.state import AgentState
from ds_agent.tools.e2b import E2BTools
from ds_agent.config import settings , Nodes
from ds_agent.utils.logger import logger 
from ds_agent.core.llm import LLMFactory

def get_llm(model_name: Optional[str] = None):
    """
    Creates a configured LLM instance using the LLMFactory.
    Returns the RAW LLM (without retry wrapper) to allow binding tools/structured output.
    """
    if model_name is None:
        model_name = settings.model_name
        
    llm_factory = LLMFactory(model_name=model_name)
    return llm_factory.create()

def get_sandbox(config: RunnableConfig) -> AsyncSandbox:
    """
    Retrieves the sandbox session from the configuration.
    """
    sandbox = config.get("configurable", {}).get("sandbox")
    if not sandbox:
        raise ValueError("Sandbox not found in config. Ensure 'sandbox' is passed in 'configurable'.")
    return sandbox

async def run_worker(state: AgentState, system_prompt: str, sender_name: str, model_name: Optional[str] = None) -> Dict[str, Any]:
    """
    Generic worker execution logic.
    
    Args:
        state: The current agent state.
        system_prompt: The persona/instructions for this worker.
        sender_name: The name of the worker (used for tracking).
        model_name: Optional model name to use for this worker.
        
    Returns:
        Dict update for the state.
    """
    logger.info(f"{sender_name} agent started")
    
    # Track node visits
    node_visits = state.get("node_visits", {}).copy()
    node_visits[sender_name] = node_visits.get(sender_name, 0) + 1
    
    if node_visits[sender_name] > settings.node_recursion_limit:
        logger.warning(f"Node {sender_name} exceeded recursion limit ({settings.node_recursion_limit}). Routing to Reporter.")
        return {
            "next": Nodes.REPORTER,
            "node_visits": node_visits,
            "messages": [SystemMessage(content=f"System: Node '{sender_name}' reached recursion limit. Terminating workflow.")]
        }

    llm = get_llm(model_name=model_name)
    
    # We instantiate tools with None just to get definitions for binding
    tool_defs = E2BTools(None).get_tools()
    llm_with_tools = llm.bind_tools(tool_defs)
    
    # Apply retries AFTER binding tools
    if settings.max_retries > 0:
        llm_with_tools = llm_with_tools.with_retry(stop_after_attempt=settings.max_retries)
    
    # Inject Supervisor Instructions if available
    instructions = state.get("supervisor_instructions", "")
    if instructions:
        system_prompt = f"{system_prompt}\n\n### MANAGER INSTRUCTIONS ###\n{instructions}"
    
    # Prepend the specialized system prompt to the message history
    current_messages = [SystemMessage(content=system_prompt)] + state['messages']
    
    response = await llm_with_tools.ainvoke(current_messages)
    return {"messages": [response], "sender": sender_name, "node_visits": node_visits}

def _prompt_to_text(prompt_value: Union[str, List[BaseMessage]]) -> str:
    """Helper to serialize a list of messages into a string for raw prompting."""
    if isinstance(prompt_value, str):
        return prompt_value
    if isinstance(prompt_value, list):
        return "\n".join([f"[{m.type.upper()}]: {m.content}" for m in prompt_value])
    return str(prompt_value)

async def invoke_structured_with_recovery(
    llm: Any,
    prompt_value: Any,
    schema_model: Type[BaseModel],
    fallback_prompt: Optional[str] = None,
) -> Tuple[BaseModel, Optional[Dict[str, str]]]:
    """
    Attempts to get structured output from the LLM. 
    If it fails, it retries with a 'fix prompt' asking for raw JSON.
    """
    try:
        # 1. Primary Attempt: Standard tool/function calling mechanism
        logger.info(f"Attempting structured output for {schema_model.__name__}...")
        chain = llm.with_structured_output(schema_model)
        out = await chain.ainvoke(prompt_value)
        
        if out is None:
            raise ValueError("LLM returned None for structured output")
            
        return out, None

    except Exception as e:
        logger.warning(f"Structured output failed ({type(e).__name__}: {e}). Attempting recovery...")
        
        prompt_text = _prompt_to_text(prompt_value)
        schema_json = schema_model.model_json_schema()
        
        # 2. Recovery Attempt: "Fix Prompt"
        fix_prompt = f"""
        You failed to provide the correct structured output.
        
        TASK: Return ONLY valid JSON matching this schema:
        {schema_json}
        
        RULES:
        - Do not output markdown code blocks (```json ... ```). 
        - Just the raw JSON string.
        - Use null when fields are unknown.

        CONTEXT:
        {prompt_text}
        """
        
        try:
            raw_msg = await llm.ainvoke(fix_prompt)
            raw = raw_msg.content if hasattr(raw_msg, "content") else str(raw_msg)
            
            # Clean common markdown wrappers
            raw_cleaned = raw.replace('```json', '').replace('```', '').strip()
            
            out = schema_model.model_validate_json(raw_cleaned)
            logger.info("Structured output recovered using Fix Prompt.")
            return out, {"recovered": "fix_prompt"}
            
        except (ValidationError, Exception) as e2:
            logger.warning(f"Recovery attempt 1 failed ({e2}). Attempting fallback...")
            
            # 3. Fallback Attempt: Strict JSON Instruction
            if fallback_prompt is None:
                fallback_prompt = f"""
                CRITICAL FAILURE RECOVERY.
                Return ONLY valid JSON matching this schema:
                {schema_json}
                """
            
            final_prompt = f"{fallback_prompt}\n\nCONTEXT:\n{prompt_text}"
            
            raw2_msg = await llm.ainvoke(final_prompt)
            raw2 = raw2_msg.content if hasattr(raw2_msg, "content") else str(raw2_msg)
            raw2_cleaned = raw2.replace('```json', '').replace('```', '').strip()
            
            out = schema_model.model_validate_json(raw2_cleaned)
            logger.info("Structured output recovered using Fallback Prompt.")
            return out, {"recovered": "json_only_fallback"}