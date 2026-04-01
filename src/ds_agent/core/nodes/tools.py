from typing import Dict, Any
import json
from langchain_core.messages import ToolMessage
from langchain_core.runnables import RunnableConfig

from ds_agent.core.state import AgentState
from ds_agent.utils.helpers import get_sandbox, update_runtime_state
from ds_agent.tools.e2b import E2BTools
from ds_agent.utils.logger import logger
from ds_agent.config import Nodes

async def tool_node(state: AgentState, config: RunnableConfig) -> Dict[str, Any]:
    """
    Executes tools requested by the LLM.
    """
    logger.info("Tool node execution")
    
    # Track node visits
    node_visits = state.get("node_visits", {}).copy()
    node_visits[Nodes.TOOLS] = node_visits.get(Nodes.TOOLS, 0) + 1
    
    sandbox = get_sandbox(config)
        
    new_cells = []
    def update_callback(cell_data):
        new_cells.append(cell_data)

    e2b_tools = E2BTools(sandbox, update_state_callback=update_callback)
    tool_map = {t.name: t for t in e2b_tools.get_tools()}
    
    last_message = state['messages'][-1]
    results = []
    runtime_state = state.get("runtime_state", {})
    producer = state.get("next", "unknown")
    
    if not hasattr(last_message, 'tool_calls'):
         logger.warning("Tool node called but last message has no tool_calls")
         return {"messages": [], "notebook_cells": []}

    for tool_call in last_message.tool_calls:
        tool_name = tool_call['name']
        tool_args = tool_call['args']
        tool_id = tool_call['id']
        
        logger.info(f"Executing tool: {tool_name}")
        if tool_name in tool_map:
            try:
                tool_instance = tool_map[tool_name]
                output = await tool_instance.ainvoke(tool_args)
            except Exception as e:
                logger.error(f"Error executing {tool_name}: {e}", exc_info=True)
                output = {
                    "ok": False,
                    "tool_name": tool_name,
                    "summary": f"خطا در اجرای ابزار {tool_name}: {str(e)}",
                    "error_type": "tool_execution_error",
                    "error_message": str(e),
                    "created_files": [],
                    "modified_files": [],
                    "produced_images": [],
                    "variables": {},
                    "text": f"خطا در اجرای ابزار {tool_name}: {str(e)}",
                }
        else:
            output = {
                "ok": False,
                "tool_name": tool_name,
                "summary": f"خطا: ابزار '{tool_name}' یافت نشد",
                "error_type": "tool_not_found",
                "error_message": f"Tool {tool_name} not found",
                "created_files": [],
                "modified_files": [],
                "produced_images": [],
                "variables": {},
                "text": f"خطا: ابزار '{tool_name}' یافت نشد",
            }

        if isinstance(output, dict):
            runtime_state = update_runtime_state(runtime_state, output, producer=producer)

        if isinstance(output, dict) and "text" in output:
            content = json.dumps(output, ensure_ascii=False)
        else:
            content = str(output)
            
        results.append(ToolMessage(tool_call_id=tool_id, name=tool_name, content=content))

    return {
        "messages": results,
        "notebook_cells": new_cells,
        "node_visits": node_visits,
        "runtime_state": runtime_state,
    }
