from typing import Literal, Dict, Any
from pydantic import BaseModel, Field
from langchain_core.messages import SystemMessage, HumanMessage

from ds_agent.core.state import AgentState
from ds_agent.utils.logger import logger
from ds_agent.config import settings, Nodes
from ds_agent.core.prompts import SUPERVISOR_PROMPT
from ds_agent.utils.helpers import build_runtime_context, get_llm, get_session_id, invoke_structured_with_recovery

# --- Models ---
class SupervisorDecision(BaseModel):
    reasoning: str = Field(description="Review of previous work and justification for the next step.")
    instructions: str = Field(description="Specific, detailed instructions for the next agent.")
    success_criteria: list[str] = Field(default_factory=list, description="Concrete checks the next agent must satisfy.")
    expected_artifacts: list[str] = Field(default_factory=list, description="Files or artifacts that should exist after the next step.")
    verification_steps: list[str] = Field(default_factory=list, description="How the next step should be verified before advancing.")
    next_agent: Literal["cleaner", "eda", "feature_engineer", "trainer", "storyteller", "reporter", "FINISH"]

async def supervisor_node(state: AgentState) -> Dict[str, Any]:
    """
    Supervisor determines which agent should act next, providing instructions.
    Uses robust recovery to ensure structured output.
    """
    logger.info("Supervisor deciding next step...")
    
    # Track node visits
    node_visits = state.get("node_visits", {}).copy()
    node_visits[Nodes.SUPERVISOR] = node_visits.get(Nodes.SUPERVISOR, 0) + 1
    
    if node_visits[Nodes.SUPERVISOR] > settings.node_recursion_limit:
        logger.warning(f"Supervisor exceeded recursion limit ({settings.node_recursion_limit}). Routing to Reporter.")
        return {
            "next": Nodes.REPORTER,
            "node_visits": node_visits,
            "messages": [SystemMessage(content="سیستم: ناظر به حد مجاز تکرار رسید. پایان دادن به جریان کاری.")]
        }

    llm = get_llm(model_name=settings.supervisor_model_name)
    
    supervisor_prompt = f"{SUPERVISOR_PROMPT}{build_runtime_context(state)}"
    messages = [SystemMessage(content=supervisor_prompt)] + state['messages']
    
    try:
        # Use the recovery helper instead of direct chain invocation
        response, metadata = await invoke_structured_with_recovery(
            llm=llm,
            prompt_value=messages,
            schema_model=SupervisorDecision,
            node_name=Nodes.SUPERVISOR,
            session_id=get_session_id(state=state),
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
            "supervisor_contract": {
                "success_criteria": response.success_criteria,
                "expected_artifacts": response.expected_artifacts,
                "verification_steps": response.verification_steps,
            },
            "node_visits": node_visits,
            # We append the Supervisor's thought process to the history so it persists
            "messages": [HumanMessage(content=(
                f"**تصمیم ناظر:**\n"
                f"*استدلال:* {response.reasoning}\n"
                f"*دستورالعمل‌ها:* {response.instructions}\n"
                f"*معیارهای موفقیت:* {response.success_criteria}\n"
                f"*خروجی‌های مورد انتظار:* {response.expected_artifacts}\n"
                f"*گام‌های راستی‌آزمایی:* {response.verification_steps}"
            ))]
        }
    except Exception as e:
        logger.error(f"Error in Supervisor node: {e}", exc_info=True)
        return {
            "next": Nodes.REPORTER,
            "node_visits": node_visits,
            "messages": [SystemMessage(content=f"ناظر با یک خطای بحرانی مواجه شد: {str(e)}. پایان دادن به جریان کاری.")]
        }
