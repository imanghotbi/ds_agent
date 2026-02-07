import operator
from typing import List, Dict, Any, TypedDict, Optional, Annotated
from langchain_core.messages import BaseMessage
from langgraph.graph.message import add_messages

class AgentState(TypedDict):
    """
    State for the Data Science Agent.
    
    Attributes:
        messages: List[BaseMessage] (Standard chat history)
        notebook_cells: List[Dict] (To track the notebook structure explicitly)
        cwd: str (Current working directory)
        next: str (Next agent to run)
    """
    # Use add_messages to append new messages to the history
    messages: Annotated[List[BaseMessage], add_messages]
    
    # Use operator.add to append new notebook cells
    notebook_cells: Annotated[List[Dict[str, Any]], operator.add]
    
    # These are overwritten (single value)
    cwd: str
    next: str
    supervisor_instructions: str