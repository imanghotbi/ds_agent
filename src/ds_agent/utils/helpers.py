import argparse
import base64
import json
import os
import re
import shlex
import tempfile
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, Any, List, Optional, Type, Union, Tuple, Iterable
from pydantic import BaseModel, ValidationError
from langchain_core.messages import SystemMessage, BaseMessage
from langchain_core.runnables import RunnableConfig
from e2b_code_interpreter import AsyncSandbox

from ds_agent.core.state import AgentState
from ds_agent.tools.e2b import E2BTools
from ds_agent.config import settings , Nodes
from ds_agent.utils.logger import logger 
from ds_agent.core.llm import LLMFactory
from ds_agent.utils.mongo_logger import mongo_llm_logger

def parse_scenario_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run ds-agent scenarios from disk.")
    parser.add_argument(
        "--scenario-dir",
        default=settings.scenario_dir,
        help="Directory containing numbered scenario folders.",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Re-run scenarios even if metadata marks them as successful.",
    )
    return parser.parse_args()

def get_llm(model_name: Optional[str] = None, **kwargs):
    """
    Creates a configured LLM instance using the LLMFactory.
    Returns the RAW LLM (without retry wrapper) to allow binding tools/structured output.
    """
    if model_name is None:
        model_name = settings.model_name
        
    llm_factory = LLMFactory(model_name=model_name, **kwargs)
    return llm_factory.create()

def get_sandbox(config: RunnableConfig) -> AsyncSandbox:
    """
    Retrieves the sandbox session from the configuration.
    """
    sandbox = config.get("configurable", {}).get("sandbox")
    if not sandbox:
        raise ValueError("Sandbox not found in config. Ensure 'sandbox' is passed in 'configurable'.")
    return sandbox

def get_session_id(state: Optional[AgentState] = None, config: Optional[RunnableConfig] = None) -> str:
    if config:
        session_id = config.get("configurable", {}).get("session_id")
        if session_id:
            return session_id
    if state:
        session_id = state.get("session_id")
        if session_id:
            return session_id
    return "system"

def build_scenario_session_id(folder_name: str) -> str:
    return f"session_id_{folder_name}_{uuid.uuid4().hex}"

def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()

def write_json_atomic(path: Path, payload: Dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile("w", encoding="utf-8", dir=path.parent, delete=False) as handle:
        json.dump(payload, handle, ensure_ascii=False, indent=2)
        handle.flush()
        os.fsync(handle.fileno())
        temp_name = handle.name
    os.replace(temp_name, path)

def write_text_atomic(path: Path, content: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile("w", encoding="utf-8", dir=path.parent, delete=False) as handle:
        handle.write(content)
        handle.flush()
        os.fsync(handle.fileno())
        temp_name = handle.name
    os.replace(temp_name, path)

def json_safe(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(key): json_safe(item) for key, item in value.items()}
    if isinstance(value, list):
        return [json_safe(item) for item in value]
    if isinstance(value, tuple):
        return [json_safe(item) for item in value]
    if isinstance(value, bytes):
        return {
            "__type__": "base64_bytes",
            "data": base64.b64encode(value).decode("ascii"),
        }
    if isinstance(value, (str, int, float, bool)) or value is None:
        return value
    return str(value)

def message_to_dict(message: BaseMessage) -> Dict[str, Any]:
    payload: Dict[str, Any] = {
        "type": message.type,
        "content": message.content,
    }
    if getattr(message, "name", None):
        payload["name"] = message.name
    if getattr(message, "tool_calls", None):
        payload["tool_calls"] = message.tool_calls
    if getattr(message, "tool_call_id", None):
        payload["tool_call_id"] = message.tool_call_id
    if getattr(message, "response_metadata", None):
        payload["response_metadata"] = message.response_metadata
    if getattr(message, "usage_metadata", None):
        payload["usage_metadata"] = message.usage_metadata
    return json_safe(payload)

def serialize_state(state: Dict[str, Any]) -> Dict[str, Any]:
    return {
        "messages": [message_to_dict(message) for message in state.get("messages", [])],
        "notebook_cells": json_safe(state.get("notebook_cells", [])),
        "cwd": state.get("cwd", "/home/user"),
        "next": state.get("next", Nodes.SUPERVISOR),
        "supervisor_instructions": state.get("supervisor_instructions", ""),
        "supervisor_contract": json_safe(state.get("supervisor_contract", {})),
        "node_visits": state.get("node_visits", {}),
        "session_id": state.get("session_id", "system"),
        "scenario_name": state.get("scenario_name", ""),
        "scenario_path": state.get("scenario_path", ""),
        "output_dir": state.get("output_dir", ""),
        "sandbox_file_manifest": state.get("sandbox_file_manifest", ""),
        "requirements": json_safe(state.get("requirements", {})),
        "runtime_state": json_safe(state.get("runtime_state", {})),
    }

def append_transcript_line(transcript: List[str], node_name: str, message: BaseMessage) -> None:
    transcript.append(f"## {node_name}")
    transcript.append(f"- type: {message.type}")
    if getattr(message, "name", None):
        transcript.append(f"- name: {message.name}")
    if getattr(message, "tool_calls", None):
        transcript.append(f"- tool_calls: {json.dumps(message.tool_calls, ensure_ascii=False)}")
    transcript.append("")
    transcript.append(str(message.content))
    transcript.append("")

def scenario_directories(root: Path) -> Iterable[Path]:
    numeric_dirs: List[Path] = []
    for child in root.iterdir():
        if child.is_dir() and child.name.isdigit():
            numeric_dirs.append(child)
    return sorted(numeric_dirs, key=lambda item: int(item.name))

def load_metadata(path: Path) -> Dict[str, Any]:
    if not path.exists():
        return {}
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return {}

async def upload_file_to_sandbox(sandbox: AsyncSandbox, local_path: Path, remote_path: str) -> None:
    remote_dir = os.path.dirname(remote_path)
    if remote_dir:
        await sandbox.commands.run(f"mkdir -p {shlex.quote(remote_dir)}", timeout=60)
    with local_path.open("rb") as handle:
        await sandbox.files.write(remote_path, handle.read())

async def upload_data_directory_to_sandbox(sandbox: AsyncSandbox, data_dir: Path) -> List[str]:
    uploaded: List[str] = []
    for file_path in sorted(path for path in data_dir.rglob("*") if path.is_file()):
        relative_path = file_path.relative_to(data_dir).as_posix()
        remote_path = f"data/{relative_path}"
        await upload_file_to_sandbox(sandbox, file_path, remote_path)
        uploaded.append(remote_path)
    return uploaded

def build_sandbox_file_manifest(uploaded_files: List[str]) -> str:
    if not uploaded_files:
        return ""
    return (
        "The following files are available under the `data/` directory in the sandbox:\n"
        + "\n".join(uploaded_files)
    )

def build_runtime_context(state: Dict[str, Any]) -> str:
    parts: List[str] = []
    file_manifest = state.get("sandbox_file_manifest", "").strip()
    if file_manifest:
        parts.append(file_manifest)
    runtime_summary = build_runtime_summary(state)
    if runtime_summary:
        parts.append(runtime_summary)
    if not parts:
        return ""
    return "\n\n### RUNTIME CONTEXT ###\n" + "\n\n".join(parts)

def extract_requirements(prompt_text: str) -> Dict[str, Any]:
    filename_matches = re.findall(
        r"\b[\w,\- ]+\.(?:csv|xlsx|xls|json|png|jpg|jpeg|svg|pdf|pkl|joblib|ipynb|txt)\b",
        prompt_text,
        flags=re.IGNORECASE,
    )
    required_filenames = sorted({match.strip() for match in filename_matches})

    prompt_lower = prompt_text.lower()
    task_type = "unknown"
    if any(token in prompt_lower for token in ("classification", "classify", "classifier")):
        task_type = "classification"
    elif any(token in prompt_lower for token in ("regression", "predict price", "predict value")):
        task_type = "regression"
    elif "clustering" in prompt_lower:
        task_type = "clustering"
    elif "time-series" in prompt_lower or "time series" in prompt_lower:
        task_type = "time-series"

    requested_stages = {
        "cleaning_only": "eda" not in prompt_lower and "train" not in prompt_lower and "model" not in prompt_lower and "feature" not in prompt_lower,
        "eda_only": "eda" in prompt_lower and "train" not in prompt_lower and "model" not in prompt_lower,
    }

    return {
        "task_type": task_type,
        "required_filenames": required_filenames,
        "requested_stages": requested_stages,
    }

def build_runtime_summary(state: Dict[str, Any]) -> str:
    runtime_state = state.get("runtime_state") or {}
    requirements = state.get("requirements") or {}

    artifacts = runtime_state.get("artifacts", [])
    variables = runtime_state.get("variables", {})
    last_execution = runtime_state.get("last_execution")
    unresolved_errors = runtime_state.get("unresolved_errors", [])

    lines: List[str] = []
    if requirements:
        lines.append(f"Requirements: {json.dumps(requirements, ensure_ascii=False)}")
    if artifacts:
        lines.append(
            "Artifacts: "
            + json.dumps(
                [
                    {
                        "path": item.get("path"),
                        "type": item.get("type"),
                        "producer": item.get("producer"),
                        "exists": item.get("exists", True),
                    }
                    for item in artifacts[-20:]
                ],
                ensure_ascii=False,
            )
        )
    if variables:
        lines.append(f"Variables: {json.dumps(variables, ensure_ascii=False)}")
    if last_execution:
        lines.append(f"Last execution: {json.dumps(last_execution, ensure_ascii=False)}")
    if unresolved_errors:
        lines.append(f"Unresolved errors: {json.dumps(unresolved_errors[-10:], ensure_ascii=False)}")
    return "\n".join(lines)

def _upsert_artifact(artifacts: List[Dict[str, Any]], artifact: Dict[str, Any]) -> List[Dict[str, Any]]:
    filtered = [item for item in artifacts if item.get("path") != artifact.get("path")]
    filtered.append(artifact)
    return filtered

def update_runtime_state(
    current_runtime_state: Optional[Dict[str, Any]],
    tool_result: Dict[str, Any],
    producer: str,
) -> Dict[str, Any]:
    runtime_state = dict(current_runtime_state or {})
    artifacts = list(runtime_state.get("artifacts", []))
    variables = dict(runtime_state.get("variables", {}))
    unresolved_errors = list(runtime_state.get("unresolved_errors", []))

    for path in tool_result.get("created_files", []):
        artifacts = _upsert_artifact(
            artifacts,
            {"path": path, "type": "file", "producer": producer, "exists": True},
        )
    for path in tool_result.get("modified_files", []):
        artifacts = _upsert_artifact(
            artifacts,
            {"path": path, "type": "file", "producer": producer, "exists": True},
        )
    for path in tool_result.get("produced_images", []):
        artifacts = _upsert_artifact(
            artifacts,
            {"path": path, "type": "image", "producer": producer, "exists": True},
        )

    for name, meta in (tool_result.get("variables", {}) or {}).items():
        variables[name] = meta

    if tool_result.get("ok"):
        unresolved_errors = []
    elif tool_result.get("error_message"):
        unresolved_errors.append(tool_result["error_message"])

    runtime_state["artifacts"] = artifacts
    runtime_state["variables"] = variables
    runtime_state["last_execution"] = {
        "tool_name": tool_result.get("tool_name"),
        "ok": tool_result.get("ok"),
        "summary": tool_result.get("summary"),
        "producer": producer,
    }
    runtime_state["unresolved_errors"] = unresolved_errors[-20:]
    return runtime_state

def normalize_relative_path(path: str) -> str:
    normalized = path.replace("\\", "/").strip()
    while normalized.startswith("./"):
        normalized = normalized[2:]
    return normalized.strip("/")

def required_file_exists(required_name: str, downloaded_files: List[str]) -> bool:
    required_normalized = normalize_relative_path(required_name)
    required_basename = os.path.basename(required_normalized)

    for downloaded in downloaded_files:
        downloaded_normalized = normalize_relative_path(downloaded)
        if downloaded_normalized == required_normalized:
            return True
        if downloaded_normalized.endswith(f"/{required_normalized}"):
            return True
        if os.path.basename(downloaded_normalized) == required_basename:
            return True

    return False

def build_initial_state(session_id: str, scenario_dir: Path, output_dir: Path) -> Dict[str, Any]:
    return {
        "messages": [],
        "notebook_cells": [],
        "cwd": "/home/user",
        "next": Nodes.SUPERVISOR,
        "supervisor_instructions": "",
        "supervisor_contract": {},
        "node_visits": {},
        "session_id": session_id,
        "scenario_name": scenario_dir.name,
        "scenario_path": str(scenario_dir),
        "output_dir": str(output_dir),
        "sandbox_file_manifest": "",
        "requirements": {},
        "runtime_state": {
            "artifacts": [],
            "variables": {},
            "last_execution": None,
            "unresolved_errors": [],
        },
    }

async def log_llm_response(node_name: str, session_id: str, response: Any) -> None:
    await mongo_llm_logger.store_llm_call_log(
        node_name=node_name,
        session_id=session_id,
        response=response,
    )

async def run_worker(state: AgentState, system_prompt: str, sender_name: str, model_name: Optional[str] = None, include_download: bool = False) -> Dict[str, Any]:
    """
    Generic worker execution logic.
    
    Args:
        state: The current agent state.
        system_prompt: The persona/instructions for this worker.
        sender_name: The name of the worker (used for tracking).
        model_name: Optional model name to use for this worker.
        include_download: Whether to allow the worker to download files (default: False).
        
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
            "messages": [SystemMessage(content=f"سیستم: عامل '{sender_name}' به حد مجاز تکرار رسید. پایان دادن به جریان کاری.")]
        }

    llm = get_llm(model_name=model_name)
    
    # We instantiate tools with None just to get definitions for binding
    tool_defs = E2BTools(None).get_tools(include_download=include_download)
    llm_with_tools = llm.bind_tools(tool_defs)
    
    # Apply retries AFTER binding tools
    if settings.max_retries > 0:
        llm_with_tools = llm_with_tools.with_retry(stop_after_attempt=settings.max_retries)
    
    # Inject Supervisor Instructions if available
    instructions = state.get("supervisor_instructions", "")
    supervisor_contract = state.get("supervisor_contract", {})
    runtime_context = build_runtime_context(state)
    if runtime_context:
        system_prompt = f"{system_prompt}{runtime_context}"
    if instructions:
        system_prompt = f"{system_prompt}\n\n### MANAGER INSTRUCTIONS ###\n{instructions}"
    if supervisor_contract:
        system_prompt = (
            f"{system_prompt}\n\n### MANAGER CONTRACT ###\n"
            f"{json.dumps(supervisor_contract, ensure_ascii=False)}"
        )
    
    # Prepend the specialized system prompt to the message history
    current_messages = [SystemMessage(content=system_prompt)] + state['messages']
    
    try:
        response = await llm_with_tools.ainvoke(current_messages)
        await log_llm_response(sender_name, get_session_id(state=state), response)
        return {"messages": [response], "sender": sender_name, "node_visits": node_visits}
    except Exception as e:
        logger.error(f"Error in node {sender_name}: {e}", exc_info=True)
        # Return a system message describing the error so the agent/supervisor is aware
        error_message = SystemMessage(content=f"خطا در اجرای {sender_name}: {str(e)}")
        return {"messages": [error_message], "sender": sender_name, "node_visits": node_visits}

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
    node_name: str,
    session_id: str,
    fallback_prompt: Optional[str] = None,
) -> Tuple[BaseModel, Optional[Dict[str, str]]]:
    """
    Attempts to get structured output from the LLM. 
    If it fails, it retries with a 'fix prompt' asking for raw JSON.
    """
    try:
        # 1. Primary Attempt: Standard tool/function calling mechanism
        logger.info(f"Attempting structured output for {schema_model.__name__}...")
        chain = llm.with_structured_output(schema_model, include_raw=True)
        out = await chain.ainvoke(prompt_value)

        parsed = out.get("parsed") if isinstance(out, dict) else out
        raw_response = out.get("raw") if isinstance(out, dict) else None
        if raw_response is not None:
            await log_llm_response(node_name, session_id, raw_response)

        if parsed is None:
            raise ValueError("LLM returned None for structured output")

        return parsed, None

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
            await log_llm_response(node_name, session_id, raw_msg)
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
            await log_llm_response(node_name, session_id, raw2_msg)
            raw2 = raw2_msg.content if hasattr(raw2_msg, "content") else str(raw2_msg)
            raw2_cleaned = raw2.replace('```json', '').replace('```', '').strip()
            
            out = schema_model.model_validate_json(raw2_cleaned)
            logger.info("Structured output recovered using Fallback Prompt.")
            return out, {"recovered": "json_only_fallback"}
