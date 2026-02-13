import os
from typing import Dict, Any
from langchain_core.messages import AIMessage
from langchain_core.runnables import RunnableConfig

from ds_agent.core.state import AgentState
from ds_agent.config import Nodes
from ds_agent.utils.helpers import get_sandbox
from ds_agent.utils.logger import logger
from ds_agent.utils.notebook import save_session_to_ipynb
from ds_agent.tools.e2b import E2BTools

async def reporter_node(state: AgentState, config: RunnableConfig) -> Dict[str, Any]:
    """
    The Reporter node handles the final export of files and notebook.
    It uses the E2BTools class to ensure consistent file handling.
    """
    logger.info("Reporter node: Finalizing results and downloading artifacts...")
    
    # Track node visits
    node_visits = state.get("node_visits", {}).copy()
    node_visits[Nodes.REPORTER] = node_visits.get(Nodes.REPORTER, 0) + 1
    
    sandbox = get_sandbox(config)
    e2b_tools = E2BTools(sandbox)
    
    # 1. Download generated files
    # We look for files created/modified during the session (excluding common system files)
    try:
        files = await sandbox.files.list(".")
        important_extensions = ['.csv', '.xlsx', '.json', '.png', '.jpg', '.pdf', '.pkl']
        
        downloaded = []
        for file in files:
            # Skip directories
            if getattr(file, 'is_dir', False):
                continue
                
            if any(file.name.endswith(ext) for ext in important_extensions):
                logger.info(f"Downloading artifact via tool: {file.name}")
                # Use the tool logic
                result = await e2b_tools.download_file(remote_path=file.name)
                if "Success" in result:
                    downloaded.append(file.name)
                else:
                    logger.error(f"Failed to download {file.name}: {result}")
                    
    except Exception as e:
        logger.error(f"Error listing/downloading artifacts: {e}")
        downloaded = []

    # 2. Export Notebook
    notebook_path = "final_analysis.ipynb"
    try:
        notebook_path = save_session_to_ipynb(state, notebook_path)
    except Exception as e:
        logger.error(f"Error exporting notebook: {e}")
        notebook_path = "Error exporting notebook"

    # 3. Create Final Summary Message
    summary = (
        "### Workflow Completed Successfully! ###\n\n"
        f"**1. Notebook Exported:** `{notebook_path}`\n"
        f"**2. Files Downloaded:** {', '.join([f'`{d}`' for d in downloaded]) if downloaded else 'None'}\n\n"
        "All variables and files are preserved in the shared sandbox for this session."
    )
    
    return {
        "messages": [AIMessage(content=summary)],
        "next": "END",
        "node_visits": node_visits
    }