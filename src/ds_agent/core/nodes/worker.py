from typing import Dict, Any

from ds_agent.core.state import AgentState
from ds_agent.utils.helpers import run_worker
from ds_agent.config import Nodes
from ds_agent.core.prompts import CLEANER_PROMPT, EDA_PROMPT, FE_PROMPT, TRAINER_PROMPT

async def cleaner_node(state: AgentState) -> Dict[str, Any]:
    """
    Data Cleaning Agent.
    """
    return await run_worker(state, CLEANER_PROMPT, Nodes.CLEANER)

async def eda_node(state: AgentState) -> Dict[str, Any]:
    """
    EDA Agent.
    """
    return await run_worker(state, EDA_PROMPT, Nodes.EDA)

async def feature_engineer_node(state: AgentState) -> Dict[str, Any]:
    """
    Feature Engineering Agent.
    """
    return await run_worker(state, FE_PROMPT, Nodes.FEATURE_ENGINEER)

async def trainer_node(state: AgentState) -> Dict[str, Any]:
    """
    Model Training Agent.
    """
    return await run_worker(state, TRAINER_PROMPT, Nodes.TRAINER)
