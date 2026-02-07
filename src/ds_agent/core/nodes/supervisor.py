from typing import Literal, Dict, Any
from pydantic import BaseModel, Field
from langchain_core.messages import SystemMessage, HumanMessage

from ds_agent.core.state import AgentState
from ds_agent.utils.helpers import get_llm
from ds_agent.utils.logger import logger
from ds_agent.core.prompts import SUPERVISOR_PROMPT
from ds_agent.utils.helpers import get_llm, invoke_structured_with_recovery

# --- Models ---
class SupervisorDecision(BaseModel):
    reasoning: str = Field(description="Review of previous work and justification for the next step.")
    instructions: str = Field(description="Specific, detailed instructions for the next agent.")
    next_agent: Literal["cleaner", "eda", "feature_engineer", "trainer", "storyteller", "reporter", "FINISH"]

async def supervisor_node(state: AgentState) -> Dict[str, Any]:
    """
    Supervisor determines which agent should act next, providing instructions.
    Uses robust recovery to ensure structured output.
    """
    logger.info("Supervisor deciding next step...")
    llm = get_llm(model_name='qwen/qwen3-235b-a22b')
    
    messages = [SystemMessage(content=SUPERVISOR_PROMPT)] + state['messages']
    
    # Use the recovery helper instead of direct chain invocation
    response, metadata = await invoke_structured_with_recovery(
        llm=llm,
        prompt_value=messages,
        schema_model=SupervisorDecision
    )
    
    next_agent = response.next_agent
    
    # Log the decision for debugging/visibility
    logger.info(f"Supervisor Reasoning: {response.reasoning}")
    logger.info(f"Supervisor Instructions: {response.instructions}")
    logger.info(f"Supervisor routed to: {next_agent}")
    
    if metadata:
        logger.info(f"Supervisor output recovered via: {metadata}")
    
    # We return the route AND the instructions to the state
    return {
        "next": next_agent,
        "supervisor_instructions": response.instructions,
        # We append the Supervisor's thought process to the history so it persists
        "messages": [HumanMessage(content=f"**Supervisor Decision:**\n*Reasoning:* {response.reasoning}\n*Instructions:* {response.instructions}")]
    }