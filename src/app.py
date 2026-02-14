import os
import chainlit as cl
from langchain_core.messages import HumanMessage, BaseMessage
from e2b_code_interpreter import AsyncSandbox

from ds_agent.core.graph import create_graph
from ds_agent.config import settings, Nodes
from ds_agent.utils.logger import logger
from ds_agent.utils.notebook import save_session_to_ipynb

# Initialize the graph once
graph = create_graph()

@cl.on_chat_start
async def start():
    """
    Initialize the E2B sandbox and setup the initial agent state.
    """
    try:
        # 1. Initialize E2B Sandbox
        sandbox = await AsyncSandbox.create(
            api_key=settings.e2b_api_key.get_secret_value(),
            timeout=settings.sandbox_timeout
        )
        cl.user_session.set("sandbox", sandbox)
        logger.info(f"E2B AsyncSandbox initialized (Timeout: {settings.sandbox_timeout}s).")

        # 2. Initial State
        state = {
            "messages": [],
            "notebook_cells": [],
            "cwd": "/home/user",
            "next": Nodes.SUPERVISOR,
            "node_visits": {}
        }
        cl.user_session.set("state", state)

        await cl.Message(content="Hello! I'm your Data Science Agent. I have a persistent E2B sandbox ready. How can I help you today? You can upload datasets using the attachment button.").send()

    except Exception as e:
        logger.error(f"Failed to initialize sandbox: {e}")
        await cl.ErrorMessage(content=f"Failed to initialize E2B sandbox: {str(e)}").send()

@cl.on_message
async def main(message: cl.Message):
    """
    Process incoming messages and run the agent graph.
    """
    state = cl.user_session.get("state")
    sandbox = cl.user_session.get("sandbox")

    if not state or not sandbox:
        await cl.ErrorMessage(content="Session not initialized properly.").send()
        return

    # 1. Handle File Uploads
    if message.elements:
        for element in message.elements:
            if isinstance(element, cl.File):
                filename = element.name
                
                await cl.Message(content=f"Uploading `{filename}` to sandbox...").send()
                
                # Write to sandbox - read from path as content might be None in some versions
                if element.path and os.path.exists(element.path):
                    with open(element.path, "rb") as f:
                        content = f.read()
                    await sandbox.files.write(filename, content)
                elif element.content:
                    await sandbox.files.write(filename, element.content)
                else:
                    await cl.ErrorMessage(content=f"Could not read content of `{filename}`").send()
                    continue
                
                # Notify state
                state["messages"].append(HumanMessage(content=f"[System: User uploaded file '{filename}']"))
                await cl.Message(content=f"Successfully uploaded `{filename}` to `{state['cwd']}`.").send()

    # 2. Process User Prompt
    state["messages"].append(HumanMessage(content=message.content))
    
    config = {
        "recursion_limit": 1000,
        "configurable": {"sandbox": sandbox}
    }

    # 3. Execute Graph and Stream results
    active_steps = {} # To track cl.Step instances by node name

    try:
        async for event in graph.astream(state, config=config):
            for node_name, value in event.items():
                # Create a step for the node if it doesn't exist
                if node_name not in active_steps:
                    step = cl.Step(name=node_name)
                    active_steps[node_name] = step
                    await step.send()
                else:
                    step = active_steps[node_name]

                # Update step content or send sub-messages
                if node_name in [Nodes.CLEANER, Nodes.EDA, Nodes.SUPERVISOR, Nodes.TRAINER, Nodes.STORYTELLER, Nodes.REPORTER, Nodes.FEATURE_ENGINEER]:
                    if "messages" in value:
                        last_msg = value["messages"][-1]
                        state["messages"].append(last_msg)
                        
                        if last_msg.content:
                            step.output = last_msg.content
                            await step.update()

                            # If it's a final summary or story, show it in the main chat clearly
                            if node_name in [Nodes.STORYTELLER, Nodes.REPORTER]:
                                await cl.Message(content=last_msg.content).send()
                        
                        # Handle Tool Calls
                        if hasattr(last_msg, 'tool_calls') and last_msg.tool_calls:
                            for tc in last_msg.tool_calls:
                                tool_step = cl.Step(name=f"Calling: {tc['name']}", parent_id=step.id)
                                args_str = str(tc['args'])
                                tool_step.output = f"```json\n{args_str}\n```"
                                await tool_step.send()

                    if "next" in value and value["next"] != Nodes.FINISH:
                        # Optional: Log routing
                        logger.debug(f"Routing to {value['next']}")

                elif node_name == Nodes.TOOLS:
                    state["messages"].extend(value["messages"])
                    state["notebook_cells"].extend(value["notebook_cells"])
                    
                    # Display tool results
                    for msg in value["messages"]:
                        # Truncate content for UI
                        display_content = msg.content
                        if len(display_content) > 3000:
                            display_content = display_content[:3000] + "\n\n... (output truncated for UI) ..."
                        
                        # Use code blocks for python/shell results
                        if msg.name in ["run_python", "run_shell"]:
                            formatted_content = f"```python\n{display_content}\n```"
                        else:
                            formatted_content = display_content

                        tool_res_step = cl.Step(name=f"Result: {msg.name}", parent_id=step.id)
                        tool_res_step.output = formatted_content
                        await tool_res_step.send()

        # Final Cleanup and Artifact Delivery
        # Find files downloaded by the Reporter
        important_extensions = ['.csv', '.xlsx', '.json', '.png', '.jpg', '.pdf', '.pkl', '.ipynb']
        files_to_send = []
        
        # Search current local directory for artifacts
        import time
        for f in os.listdir("."):
            if any(f.endswith(ext) for ext in important_extensions):
                # Basic heuristic: files modified in the last 5 minutes are likely artifacts
                if time.time() - os.path.getmtime(f) < 300:
                    files_to_send.append(cl.File(path=f, name=f))
        
        if files_to_send:
            await cl.Message(content="### Download Session Artifacts ###", elements=files_to_send).send()

    except Exception as e:
        logger.error(f"Error during graph execution: {e}", exc_info=True)
        await cl.ErrorMessage(content=f"An error occurred: {str(e)}").send()

@cl.on_chat_end
async def end(*args):
    """
    Cleanup sandbox and export notebook on session end.
    """
    state = cl.user_session.get("state")
    sandbox = cl.user_session.get("sandbox")

    if state and state.get("notebook_cells"):
        try:
            filename = save_session_to_ipynb(state, "chainlit_analysis.ipynb")
            await cl.Message(content=f"Session exported to `{filename}`").send()
        except Exception as e:
            logger.error(f"Failed to export notebook: {e}")

    if sandbox:
        await sandbox.kill()
        logger.info("E2B Sandbox closed.")