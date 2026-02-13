import base64
import json
import os
from typing import List, Optional, Dict, Any, Union, Tuple
from pydantic import BaseModel, Field
from langchain_core.tools import tool, StructuredTool
from e2b_code_interpreter import AsyncSandbox

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

    async def run_python(self, code: str) -> Union[str, Dict[str, Any]]:
        """
        Executes Python code in a persistent Jupyter kernel.
        Captures stdout, stderr, and images (plots).
        """
        try:
            execution = await self.sandbox.run_code(code)
            
            outputs, logs = self._process_logs(execution.logs)
            artifacts, media_outputs, text_results = self._process_results(execution.results)
            
            outputs.extend(media_outputs)
            # Add the text results (last expression values) to the logs returned to the LLM
            logs.extend(text_results)
            
            if execution.error:
                error_output, error_msg = self._process_error(execution.error)
                logs.append(f"Error: {error_msg}")
                outputs.append(error_output)

            cell_data = {
                'cell_type': 'code',
                'source': code,
                'outputs': outputs,
                'execution_count': None
            }

            if self.update_state_callback:
                self.update_state_callback(cell_data)

            response_text = self._format_response(logs, artifacts, execution.error)
            
            # Filter for image outputs to return to the LLM
            images = [o for o in outputs if o.get('type') == 'image']
            
            if images:
                return {
                    "text": response_text,
                    "images": images
                }
            return response_text

        except Exception as e:
            return f"Status: Error\nOutput: System Error - {str(e)}"

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

    async def run_shell(self, command: str) -> str:
        try:
            result = await self.sandbox.commands.run(command, timeout=60) 
            output = f"stdout: {result.stdout}\nstderr: {result.stderr}"
            if result.error:
                 output += f"\nError: {result.error}"
            return f"Status: Success\nOutput: {output}"
        except Exception as e:
            return f"Status: Error\nOutput: System Error - {str(e)}"

    async def download_file(self, remote_path: str, local_filename: Optional[str] = None) -> str:
        """
        Downloads a file from the sandbox to the local filesystem.
        """
        try:
            if not local_filename:
                local_filename = remote_path.split('/')[-1]
            
            # Use download_file for reliable binary retrieval
            content = await self.sandbox.download_file(remote_path)
            
            # Always write as binary to prevent corruption of images/pickles
            with open(local_filename, 'wb') as f:
                f.write(content)
                
            return f"Status: Success\nFile downloaded successfully to: {os.path.abspath(local_filename)}"
        except Exception as e:
            return f"Status: Error\nOutput: Failed to download file - {str(e)}"

    async def create_markdown(self, content: str) -> str:
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
        return "Status: Success\nMarkdown cell added to the notebook."

    def get_tools(self) -> List[StructuredTool]:
        return [
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
            ),
            StructuredTool.from_function(
                coroutine=self.download_file,
                name="download_file",
                description="Downloads a file from the sandbox to the local machine. Use this to provide the user with final data files, reports, or generated assets.",
                args_schema=DownloadFileInput
            )
        ]