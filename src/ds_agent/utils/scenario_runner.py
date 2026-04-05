from __future__ import annotations

import asyncio
import time
from pathlib import Path
from typing import Any, Dict, List

from e2b_code_interpreter import AsyncSandbox
from langchain_core.messages import HumanMessage

from ds_agent.config import settings
from ds_agent.core.graph import create_graph
from ds_agent.utils.helpers import (
    append_transcript_line,
    build_initial_state,
    build_sandbox_file_manifest,
    build_scenario_session_id,
    extract_requirements,
    load_metadata,
    parse_scenario_args,
    scenario_directories,
    serialize_state,
    upload_data_directory_to_sandbox,
    utc_now,
    write_json_atomic,
    write_text_atomic,
)
from ds_agent.utils.logger import logger
from ds_agent.utils.mongo_logger import mongo_llm_logger


class ScenarioRunner:
    def __init__(self, scenario_root: Path, force: bool = False) -> None:
        self.scenario_root = scenario_root
        self.force = force
        self.graph = create_graph()

    def _clear_previous_outputs(self, output_dir: Path) -> None:
        output_dir.mkdir(parents=True, exist_ok=True)

        sandbox_artifacts_dir = output_dir / "sandbox_artifacts"
        if sandbox_artifacts_dir.exists():
            for path in sorted(sandbox_artifacts_dir.rglob("*"), reverse=True):
                if path.is_file():
                    path.unlink()
                elif path.is_dir():
                    path.rmdir()
            sandbox_artifacts_dir.rmdir()

        notebook_path = output_dir / "final_analysis.ipynb"
        if notebook_path.exists():
            notebook_path.unlink()

    async def run_single_scenario(self, scenario_dir: Path) -> Dict[str, Any]:
        output_dir = scenario_dir / "output"
        metadata_path = output_dir / "metadata.json"
        state_path = output_dir / "state.json"
        transcript_path = output_dir / "transcript.md"
        prompt_path = scenario_dir / "prompt.md"
        data_dir = scenario_dir / "data"
        session_id = build_scenario_session_id(scenario_dir.name)

        existing_metadata = load_metadata(metadata_path)
        if existing_metadata.get("status") == "successful" and not self.force:
            logger.info(f"Skipping scenario {scenario_dir.name}; already successful.")
            return {"scenario": scenario_dir.name, "status": "skipped"}

        self._clear_previous_outputs(output_dir)

        start_time = time.monotonic()
        transcript_lines: List[str] = [f"# Scenario {scenario_dir.name}", "", f"- session_id: {session_id}", ""]
        state = build_initial_state(session_id, scenario_dir, output_dir)
        metadata: Dict[str, Any] = {
            "scenario": scenario_dir.name,
            "session_id": session_id,
            "status": "in_progress",
            "started_at": utc_now(),
            "completed_at": None,
            "duration_seconds": 0.0,
            "error": None,
            "prompt_path": str(prompt_path),
            "data_dir": str(data_dir) if data_dir.exists() else None,
            "uploaded_files": [],
            "output_dir": str(output_dir),
        }

        write_json_atomic(metadata_path, metadata)
        write_json_atomic(state_path, serialize_state(state))
        write_text_atomic(transcript_path, "\n".join(transcript_lines))

        if not prompt_path.exists():
            metadata["status"] = "error"
            metadata["error"] = "prompt.md not found"
            metadata["completed_at"] = utc_now()
            metadata["duration_seconds"] = round(time.monotonic() - start_time, 3)
            write_json_atomic(metadata_path, metadata)
            return {"scenario": scenario_dir.name, "status": "error", "error": metadata["error"]}

        prompt_text = prompt_path.read_text(encoding="utf-8").strip()
        if not prompt_text:
            metadata["status"] = "error"
            metadata["error"] = "prompt.md is empty"
            metadata["completed_at"] = utc_now()
            metadata["duration_seconds"] = round(time.monotonic() - start_time, 3)
            write_json_atomic(metadata_path, metadata)
            return {"scenario": scenario_dir.name, "status": "error", "error": metadata["error"]}

        try:
            async with await AsyncSandbox.create(
                api_key=settings.e2b_api_key.get_secret_value(),
                timeout=settings.sandbox_timeout,
                template=settings.sandbox_template
            ) as sandbox:
                uploaded_files: List[str] = []
                if data_dir.exists() and data_dir.is_dir():
                    uploaded_files = await upload_data_directory_to_sandbox(sandbox, data_dir)
                    state["sandbox_file_manifest"] = build_sandbox_file_manifest(uploaded_files)

                metadata["uploaded_files"] = uploaded_files
                write_json_atomic(metadata_path, metadata)

                state["messages"].append(HumanMessage(content=prompt_text))
                state["requirements"] = extract_requirements(prompt_text)
                write_json_atomic(state_path, serialize_state(state))

                config = {
                    "recursion_limit": 1000,
                    "configurable": {
                        "sandbox": sandbox,
                        "session_id": session_id,
                        "output_dir": str(output_dir),
                    },
                }

                async for event in self.graph.astream(state, config=config):
                    for node_name, value in event.items():
                        if "messages" in value:
                            for message in value["messages"]:
                                state["messages"].append(message)
                                append_transcript_line(transcript_lines, node_name, message)

                        if "notebook_cells" in value:
                            state["notebook_cells"].extend(value["notebook_cells"])

                        for key in ("next", "node_visits", "supervisor_instructions", "supervisor_contract", "runtime_state", "sender"):
                            if key in value:
                                state[key] = value[key]

                        metadata["last_node"] = node_name
                        metadata["duration_seconds"] = round(time.monotonic() - start_time, 3)
                        write_json_atomic(state_path, serialize_state(state))
                        write_text_atomic(transcript_path, "\n".join(transcript_lines))
                        write_json_atomic(metadata_path, metadata)

                final_qa = (state.get("runtime_state") or {}).get("final_qa", {})
                metadata["status"] = "successful" if final_qa.get("passed", True) else "incomplete"
                if not final_qa.get("passed", True):
                    metadata["error"] = (
                        "Final QA failed: "
                        + ", ".join(final_qa.get("missing_required", []) or final_qa.get("unresolved_errors", []) or ["unknown issue"])
                    )
                metadata["completed_at"] = utc_now()
                metadata["duration_seconds"] = round(time.monotonic() - start_time, 3)
                write_json_atomic(state_path, serialize_state(state))
                write_text_atomic(transcript_path, "\n".join(transcript_lines))
                write_json_atomic(metadata_path, metadata)
                return {"scenario": scenario_dir.name, "status": "successful"}

        except Exception as exc:
            logger.error(f"Scenario {scenario_dir.name} failed: {exc}", exc_info=True)
            metadata["status"] = "error"
            metadata["error"] = str(exc)
            metadata["completed_at"] = utc_now()
            metadata["duration_seconds"] = round(time.monotonic() - start_time, 3)
            write_json_atomic(state_path, serialize_state(state))
            write_text_atomic(transcript_path, "\n".join(transcript_lines))
            write_json_atomic(metadata_path, metadata)
            return {"scenario": scenario_dir.name, "status": "error", "error": str(exc)}

    async def run(self) -> List[Dict[str, Any]]:
        if not self.scenario_root.exists():
            raise FileNotFoundError(f"Scenario directory not found: {self.scenario_root}")
        if not self.scenario_root.is_dir():
            raise NotADirectoryError(f"Scenario path is not a directory: {self.scenario_root}")

        results: List[Dict[str, Any]] = []
        for scenario_dir in scenario_directories(self.scenario_root):
            results.append(await self.run_single_scenario(scenario_dir))
        return results


async def async_main() -> int:
    args = parse_scenario_args()
    runner = ScenarioRunner(
        scenario_root=Path(args.scenario_dir).expanduser().resolve(),
        force=args.force,
    )

    try:
        results = await runner.run()
        for result in results:
            print(f"{result['scenario']}: {result['status']}")
            if result.get("error"):
                print(f"  error: {result['error']}")
        return 0 if all(result["status"] in {"successful", "skipped"} for result in results) else 1
    finally:
        await mongo_llm_logger.close()


def main() -> int:
    return asyncio.run(async_main())
