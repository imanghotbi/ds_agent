import hashlib
import json
import os
from typing import List, Optional, Dict, Any, Union, Tuple
from pydantic import BaseModel, Field
from langchain_core.tools import tool, StructuredTool
from e2b_code_interpreter import AsyncSandbox

from ds_agent.config import settings
from ds_agent.utils.logger import logger

class RunPythonInput(BaseModel):
    code: str = Field(description="The Python code to execute.")

class RunShellInput(BaseModel):
    command: str = Field(description="The shell command to execute.")

class CreateMarkdownInput(BaseModel):
    content: str = Field(description="The markdown content to add to the notebook. Use this for titles, explanations, and summarizing findings in the generated notebook.")

class DownloadFileInput(BaseModel):
    remote_path: str = Field(description="The absolute path to the file in the sandbox (e.g., '/home/user/cleaned_data.csv').")
    local_filename: Optional[str] = Field(description="The name to save the file as locally. If not provided, the remote filename will be used.", default=None)

class E2BTools:
    def __init__(self, sandbox: AsyncSandbox, update_state_callback: Optional[callable] = None):
        """
        Args:
            sandbox: The active E2B AsyncSandbox instance.
            update_state_callback: A function to call to update the global/agent state.
        """
        self.sandbox = sandbox
        self.update_state_callback = update_state_callback

    async def _snapshot_files(self) -> Dict[str, Any]:
        try:
            return {f.name: f.modified_time for f in await self.sandbox.files.list(".")}
        except Exception:
            return {}

    async def _snapshot_variables(self) -> Dict[str, Any]:
        try:
            inspection = await self.sandbox.run_code(
                """
import json

summary = {}
for name, value in globals().items():
    if name.startswith("_"):
        continue
    if name in {"json", "summary"}:
        continue
    type_name = type(value).__name__
    meta = {"type": type_name}
    try:
        shape = getattr(value, "shape", None)
        if shape is not None:
            meta["shape"] = list(shape)
    except Exception:
        pass
    summary[name] = meta

print(json.dumps(summary, ensure_ascii=False))
"""
            )
            if inspection.logs.stdout:
                return json.loads("\n".join(inspection.logs.stdout))
        except Exception:
            pass
        return {}

    async def run_python(self, code: str) -> Dict[str, Any]:
        """
        Executes Python code in a persistent Jupyter kernel.
        Captures stdout, stderr, and images (plots).

        KEY DESIGN: Image outputs stored in cell_data use the raw bytes read directly from
        the sandbox file system (not Jupyter's inline capture). This guarantees the MD5 hash
        of a notebook-cell image is identical to the hash produced by sandbox.files.read(),
        which is what downstream exports use.
        """
        try:
            initial_files = await self._snapshot_files()

            execution = await self.sandbox.run_code(code)

            # Process logs first so `logs` is defined before we append to it
            outputs, logs = self._process_logs(execution.logs)

            # Detect new/updated image files; read their bytes once for both local
            # save and notebook-cell output (ensures byte-level consistency).
            file_image_outputs: List[Dict[str, Any]] = []
            created_files: List[str] = []
            modified_files: List[str] = []
            produced_images: List[str] = []
            image_exts = ('.png', '.jpg', '.jpeg', '.svg')
            try:
                final_files = await self.sandbox.files.list(".")
                for f in final_files:
                    is_new = f.name not in initial_files
                    is_updated = not is_new and f.modified_time > initial_files[f.name]
                    if is_new:
                        created_files.append(f.name)
                    elif is_updated:
                        modified_files.append(f.name)
                    if (is_new or is_updated) and f.name.lower().endswith(image_exts):
                        logger.info(f"Detected {'new' if is_new else 'updated'} image file: {f.name}.")
                        file_bytes = await self.sandbox.files.read(f.name, format="bytes")
                        produced_images.append(f.name)
                        file_image_outputs.append({
                            "type": "image",
                            "data": file_bytes,       # raw bytes — NOT base64
                            "mime_type": "image/png",
                            "filename": f.name,       # used for filename-based dedup in app.py
                        })
                        logs.append(f"System: Detected generated image file {f.name}.")
            except Exception as e:
                logger.warning(f"Failed to inspect new/updated files: {e}")

            # Process execution results (text + inline Jupyter image captures)
            _, media_outputs, text_results = self._process_results(execution.results)

            if file_image_outputs:
                # Prefer file-based image outputs so hashes are consistent with
                # sandbox.files.read() calls made later in get_images_from_markdown.
                outputs.extend(file_image_outputs)
            else:
                # No files written to disk — fall back to Jupyter inline captures
                outputs.extend(media_outputs)

            logs.extend(text_results)

            if execution.error:
                error_output, error_msg = self._process_error(execution.error)
                logs.append(f"Error: {error_msg}")
                outputs.append(error_output)

            cell_data = {
                'cell_type': 'code',
                'source': code,
                'outputs': outputs,
                'execution_count': None,
            }

            if self.update_state_callback:
                self.update_state_callback(cell_data)

            response_text = self._format_response(logs, [], execution.error)
            variables = await self._snapshot_variables()

            return {
                "ok": execution.error is None,
                "tool_name": "run_python",
                "summary": response_text,
                "stdout": "\n".join(execution.logs.stdout) if execution.logs.stdout else "",
                "stderr": "\n".join(execution.logs.stderr) if execution.logs.stderr else "",
                "error_type": execution.error.name if execution.error else None,
                "error_message": str(execution.error.value) if execution.error else None,
                "created_files": created_files,
                "modified_files": modified_files,
                "produced_images": produced_images,
                "variables": variables,
                "text": response_text,
            }

        except Exception as e:
            return {
                "ok": False,
                "tool_name": "run_python",
                "summary": f"Status: Error\nOutput: System Error - {str(e)}",
                "stdout": "",
                "stderr": "",
                "error_type": "system_error",
                "error_message": str(e),
                "created_files": [],
                "modified_files": [],
                "produced_images": [],
                "variables": {},
                "text": f"Status: Error\nOutput: System Error - {str(e)}",
            }

    def _process_logs(self, logs_obj) -> Tuple[List[Dict[str, Any]], List[str]]:
        outputs = []
        log_lines = []
        
        if logs_obj.stdout:
            stdout_str = "\n".join(logs_obj.stdout)
            log_lines.append(f"stdout: {stdout_str}")
            outputs.append({'type': 'stdout', 'text': stdout_str})
        
        if logs_obj.stderr:
            stderr_str = "\n".join(logs_obj.stderr)
            log_lines.append(f"stderr: {stderr_str}")
            outputs.append({'type': 'stderr', 'text': stderr_str})
            
        return outputs, log_lines

    def _process_results(self, results: List[Any]) -> Tuple[List[str], List[Dict[str, Any]], List[str]]:
        artifacts = []
        outputs = []
        text_results = []
        seen_image_hashes = set()
        
        for result in results:
            data = None
            mime_type = None
            
            if result.png:
                data = result.png
                mime_type = 'image/png'
            elif result.jpeg:
                data = result.jpeg
                mime_type = 'image/jpeg'
            elif result.svg:
                data = result.svg
                mime_type = 'image/svg+xml'
            elif result.text:
                text_results.append(result.text)
                outputs.append({'type': 'result', 'data': {'text/plain': result.text}})
                continue
            
            if data and mime_type:
                # Deduplicate within the same result set
                img_hash = hashlib.md5(data.encode() if isinstance(data, str) else data).hexdigest()
                if img_hash in seen_image_hashes:
                    continue
                seen_image_hashes.add(img_hash)

                outputs.append({
                    'type': 'image',
                    'data': data,
                    'mime_type': mime_type
                })
                
        return artifacts, outputs, text_results

    def _process_error(self, error_obj) -> Tuple[Dict[str, Any], str]:
        error_msg = f"{error_obj.name}: {error_obj.value}\n{error_obj.traceback}"
        error_output = {
            'type': 'error',
            'ename': error_obj.name,
            'evalue': error_obj.value,
            'traceback': error_obj.traceback.split('\n')
        }
        return error_output, error_msg

    def _format_response(self, logs: List[str], artifacts: List[str], error_obj) -> str:
        if error_obj:
            return f"Status: Error\nOutput: {chr(10).join(logs)}"
        return f"Status: Success\nOutput: {chr(10).join(logs)}\nArtifacts: {artifacts}"

    async def run_shell(self, command: str) -> Dict[str, Any]:
        try:
            initial_files = await self._snapshot_files()
            result = await self.sandbox.commands.run(command, timeout=300) 
            final_files = await self._snapshot_files()
            created_files = sorted(set(final_files) - set(initial_files))
            modified_files = sorted(
                path for path in final_files
                if path in initial_files and final_files[path] > initial_files[path]
            )
            ok = result.error is None
            output = f"stdout: {result.stdout}\nstderr: {result.stderr}"
            if result.error:
                 output += f"\nError: {result.error}"
            return {
                "ok": ok,
                "tool_name": "run_shell",
                "summary": f"Status: {'Success' if ok else 'Error'}\nOutput: {output}",
                "stdout": result.stdout,
                "stderr": result.stderr,
                "error_type": "command_error" if result.error else None,
                "error_message": str(result.error) if result.error else None,
                "created_files": created_files,
                "modified_files": modified_files,
                "produced_images": [path for path in created_files + modified_files if path.lower().endswith(('.png', '.jpg', '.jpeg', '.svg'))],
                "variables": {},
                "text": f"Status: {'Success' if ok else 'Error'}\nOutput: {output}",
            }
        except Exception as e:
            return {
                "ok": False,
                "tool_name": "run_shell",
                "summary": f"Status: Error\nOutput: System Error - {str(e)}",
                "stdout": "",
                "stderr": "",
                "error_type": "system_error",
                "error_message": str(e),
                "created_files": [],
                "modified_files": [],
                "produced_images": [],
                "variables": {},
                "text": f"Status: Error\nOutput: System Error - {str(e)}",
            }

    async def download_file(self, remote_path: str, local_filename: Optional[str] = None) -> Dict[str, Any]:
        """
        Downloads a file from the sandbox to the local filesystem.
        """
        try:
            if not local_filename:
                local_filename = remote_path.split('/')[-1]
            
            # Ensure local artifacts directory exists
            os.makedirs(settings.local_artifacts_dir, exist_ok=True)
            
            # Use settings.local_artifacts_dir as the base directory
            local_filepath = os.path.join(settings.local_artifacts_dir, local_filename)

            # Use sandbox.files.read with format="bytes" for reliable binary retrieval in SDK v2
            content = await self.sandbox.files.read(remote_path, format="bytes")
            
            # Always write as binary to prevent corruption of images/pickles
            with open(local_filepath, 'wb') as f:
                f.write(content)
                
            return {
                "ok": True,
                "tool_name": "download_file",
                "summary": f"Status: Success\nFile downloaded successfully to: {os.path.abspath(local_filepath)}",
                "stdout": "",
                "stderr": "",
                "error_type": None,
                "error_message": None,
                "created_files": [local_filepath],
                "modified_files": [],
                "produced_images": [],
                "variables": {},
                "text": f"Status: Success\nFile downloaded successfully to: {os.path.abspath(local_filepath)}",
            }
        except Exception as e:
            return {
                "ok": False,
                "tool_name": "download_file",
                "summary": f"Status: Error\nOutput: Failed to download file - {str(e)}",
                "stdout": "",
                "stderr": "",
                "error_type": "download_error",
                "error_message": str(e),
                "created_files": [],
                "modified_files": [],
                "produced_images": [],
                "variables": {},
                "text": f"Status: Error\nOutput: Failed to download file - {str(e)}",
            }

    async def create_markdown(self, content: str) -> Dict[str, Any]:
        """
        Adds a markdown cell to the notebook.
        """
        cell_data = {
            'cell_type': 'markdown',
            'source': content,
            'outputs': []
        }
        if self.update_state_callback:
            self.update_state_callback(cell_data)
        return {
            "ok": True,
            "tool_name": "create_markdown",
            "summary": "Status: Success\nMarkdown cell added to the notebook.",
            "stdout": "",
            "stderr": "",
            "error_type": None,
            "error_message": None,
            "created_files": [],
            "modified_files": [],
            "produced_images": [],
            "variables": {},
            "text": "Status: Success\nMarkdown cell added to the notebook.",
        }

    def get_tools(self, include_download: bool = True) -> List[StructuredTool]:
            tools = [
                StructuredTool.from_function(
                    coroutine=self.run_python,
                    name="run_python",
                    description="Executes Python code in a persistent Jupyter kernel. Use this for data analysis, visualization, and variable definition.",
                    args_schema=RunPythonInput
                ),
                StructuredTool.from_function(
                    coroutine=self.create_markdown,
                    name="create_markdown",
                    description="Adds a markdown cell to the notebook. Use this for adding titles, descriptions, and analysis narratives to the final .ipynb file.",
                    args_schema=CreateMarkdownInput
                ),
                StructuredTool.from_function(
                    coroutine=self.run_shell,
                    name="run_shell",
                    description="Executes a shell command (e.g., pip install, ls, unzip). Use this for system operations.",
                    args_schema=RunShellInput
                )
            ]

            if include_download:
                tools.append(
                    StructuredTool.from_function(
                        coroutine=self.download_file,
                        name="download_file",
                        description="Downloads a file from the sandbox to the local machine. Use this to provide the user with final data files, reports, or generated assets.",
                        args_schema=DownloadFileInput
                    )

                )

            return tools
