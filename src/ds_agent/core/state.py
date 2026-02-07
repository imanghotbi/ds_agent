import operator
from typing import List, Dict, Any, TypedDict, Optional, Annotated
from langchain_core.messages import BaseMessage
from langgraph.graph.message import add_messages
from e2b_code_interpreter import AsyncSandbox

class AgentState(TypedDict):
    """
    State for the Data Science Agent.
    
    Attributes:
        messages: List[BaseMessage] (Standard chat history)
        notebook_cells: List[Dict] (To track the notebook structure explicitly)
        sandbox_session: AsyncSandbox (The active E2B connection)
        cwd: str (Current working directory)
    """
    # Use add_messages to append new messages to the history
    messages: Annotated[List[BaseMessage], add_messages]
    
    # Use operator.add to append new notebook cells
    notebook_cells: Annotated[List[Dict[str, Any]], operator.add]
    
    # These are overwritten (single value)
    sandbox_session: Optional[AsyncSandbox]
    cwd: str
