import operator
from typing import List, Dict, Any, TypedDict, Optional, Annotated, TYPE_CHECKING
from langchain_core.messages import BaseMessage
from langgraph.graph.message import add_messages

if TYPE_CHECKING:
    from e2b_code_interpreter import AsyncSandbox

class AgentState(TypedDict):
    """
    State for the Data Science Agent.
    """
    # Use add_messages to append new messages to the history
    messages: Annotated[List[BaseMessage], add_messages]
    
    # Use operator.add to append new notebook cells
    notebook_cells: Annotated[List[Dict[str, Any]], operator.add]
    
    # These are overwritten (single value)
    sandbox_session: Optional["AsyncSandbox"]
    cwd: str