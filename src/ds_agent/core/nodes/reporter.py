import os
from typing import Dict, Any
from langchain_core.messages import AIMessage
from langchain_core.runnables import RunnableConfig

from ds_agent.core.state import AgentState
from ds_agent.config import Nodes
from ds_agent.utils.helpers import get_sandbox, required_file_exists
from ds_agent.utils.logger import logger
from ds_agent.utils.notebook import save_session_to_ipynb


async def _download_sandbox_files(sandbox, destination_dir: str) -> list[str]:
    os.makedirs(destination_dir, exist_ok=True)
    downloaded: list[str] = []

    try:
        result = await sandbox.commands.run("find . -type f | sort", timeout=120)
        if result.error:
            raise RuntimeError(result.error)

        for raw_path in result.stdout.splitlines():
            relative_path = raw_path.strip()
            if not relative_path:
                continue
            if relative_path.startswith("./"):
                relative_path = relative_path[2:]
            if not relative_path:
                continue
            path_parts = [part for part in relative_path.split("/") if part]
            if any(part.startswith(".") for part in path_parts):
                continue

            local_path = os.path.join(destination_dir, relative_path)
            os.makedirs(os.path.dirname(local_path), exist_ok=True)
            content = await sandbox.files.read(relative_path, format="bytes")
            with open(local_path, "wb") as handle:
                handle.write(content)
            downloaded.append(relative_path)
    except Exception as exc:
        logger.error(f"Error downloading sandbox artifacts: {exc}")

    return downloaded


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
    output_dir = state.get("output_dir") or config.get("configurable", {}).get("output_dir")
    artifacts_dir = os.path.join(output_dir, "sandbox_artifacts") if output_dir else "sandbox_artifacts"
    
    downloaded = await _download_sandbox_files(sandbox, artifacts_dir)
    # 2. Export Notebook
    notebook_path = os.path.join(output_dir, "final_analysis.ipynb") if output_dir else "final_analysis.ipynb"
    try:
        notebook_path = save_session_to_ipynb(state, notebook_path)
    except Exception as e:
        logger.error(f"Error exporting notebook: {e}")
        notebook_path = "Error exporting notebook"

    requirements = state.get("requirements", {})
    runtime_state = state.get("runtime_state", {})
    required_filenames = requirements.get("required_filenames", [])
    missing_required = [
        name for name in required_filenames
        if not required_file_exists(name, downloaded)
    ]
    unresolved_errors = runtime_state.get("unresolved_errors", [])
    notebook_ok = os.path.exists(notebook_path)
    qa_passed = notebook_ok and not missing_required and not unresolved_errors

    checklist_lines = [
        f"- notebook_created: {'yes' if notebook_ok else 'no'}",
        f"- required_filenames: {', '.join(required_filenames) if required_filenames else 'none explicitly requested'}",
        f"- missing_required: {', '.join(missing_required) if missing_required else 'none'}",
        f"- unresolved_errors: {', '.join(unresolved_errors) if unresolved_errors else 'none'}",
    ]

    # 3. Create Final Summary Message
    summary = (
        f"### {'جریان کار با موفقیت به اتمام رسید' if qa_passed else 'جریان کار کامل نیست'} ###\n\n"
        f"**1. نوت بوک ایجاد شده:** `{os.path.basename(notebook_path)}`\n"
        f"**2. فایل‌های دانلودشده:** {', '.join([f'`{d}`' for d in downloaded]) if downloaded else 'هیچکدام'}\n"
        f"**3. وضعیت QA:** {'PASS' if qa_passed else 'FAIL'}\n\n"
        "### چک‌لیست نهایی\n"
        + "\n".join(checklist_lines)
        + "\n"
    )
    
    return {
        "messages": [AIMessage(content=summary)],
        "next": "END",
        "node_visits": node_visits,
        "runtime_state": {
            **runtime_state,
            "final_qa": {
                "passed": qa_passed,
                "missing_required": missing_required,
                "unresolved_errors": unresolved_errors,
                "downloaded_files": downloaded,
                "notebook_path": notebook_path,
            },
        },
    }
