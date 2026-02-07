import base64
import json
from typing import List, Optional, Dict, Any, Union, Tuple
from pydantic import BaseModel, Field
from langchain_core.tools import tool, StructuredTool
from e2b_code_interpreter import AsyncSandbox

class RunPythonInput(BaseModel):
    code: str = Field(description="The Python code to execute.")

class RunShellInput(BaseModel):
    command: str = Field(description="The shell command to execute.")

class E2BTools:
    def __init__(self, sandbox: AsyncSandbox, update_state_callback: Optional[callable] = None):
        """
        Args:
            sandbox: The active E2B AsyncSandbox instance.
            update_state_callback: A function to call to update the global/agent state.
        """
        self.sandbox = sandbox
        self.update_state_callback = update_state_callback

    async def run_python(self, code: str) -> str:
        """
        Executes Python code in a persistent Jupyter kernel.
        Captures stdout, stderr, and images (plots).
        """
        try:
            execution = await self.sandbox.run_code(code)
            
            outputs, logs = self._process_logs(execution.logs)
            artifacts, media_outputs = self._process_results(execution.results)
            outputs.extend(media_outputs)
            
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

            return self._format_response(logs, artifacts, execution.error)

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

    def _process_results(self, results: List[Any]) -> Tuple[List[str], List[Dict[str, Any]]]:
        artifacts = []
        outputs = []
        
        for result in results:
            data = None
            mime_type = None
            
            if result.png:
                data = result.png
                mime_type = 'image/png'
                artifacts.append("image.png")
            elif result.jpeg:
                data = result.jpeg
                mime_type = 'image/jpeg'
                artifacts.append("image.jpeg")
            elif result.svg:
                data = result.svg
                mime_type = 'image/svg+xml'
                artifacts.append("image.svg")
            elif result.text:
                outputs.append({'type': 'result', 'data': {'text/plain': result.text}})
                continue
            
            if data and mime_type:
                outputs.append({
                    'type': 'image',
                    'data': data,
                    'mime_type': mime_type
                })
                
        return artifacts, outputs

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

    def get_tools(self) -> List[StructuredTool]:
        return [
            StructuredTool.from_function(
                coroutine=self.run_python,
                name="run_python",
                description="Executes Python code in a persistent Jupyter kernel. Use this for data analysis, visualization, and variable definition.",
                args_schema=RunPythonInput
            ),
            StructuredTool.from_function(
                coroutine=self.run_shell,
                name="run_shell",
                description="Executes a shell command (e.g., pip install, ls, unzip). Use this for system operations.",
                args_schema=RunShellInput
            )
        ]