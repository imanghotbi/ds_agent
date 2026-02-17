import os
import chainlit as cl
from langchain_core.messages import HumanMessage
from e2b_code_interpreter import AsyncSandbox
import base64
import hashlib

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
        session_id = cl.context.session.id
        logger.info(f"New chat session started. Session ID: {session_id}")
    except Exception as e:
        logger.warning(f"Could not get session ID on start: {e}")
    
    try:
        # Initialize hash tracking for images
        cl.user_session.set("displayed_image_hashes", set())

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

        await cl.Message(content="سلام! من دستیار علم داده شما هستم. یک محیط مجازی E2B پایدار برای شما آماده است. چطور می‌توانم امروز کمکتان کنم؟ می‌توانید با استفاده از دکمه پیوست، مجموعه داده‌های خود را آپلود کنید.").send()

    except Exception as e:
        logger.error(f"Failed to initialize sandbox: {e}")
        await cl.ErrorMessage(content=f"خطا در راه‌اندازی محیط مجازی E2B: {str(e)}").send()

async def get_images_from_markdown(content: str, sandbox):
    """
    Scans markdown for local image references, downloads them from the sandbox,
    and returns a list of cl.Image elements.
    Uses a session-based cache to avoid redundant downloads.
    """
    import re
    # Matches ![alt](path), ![alt]( <path> ), etc.
    pattern = r"!\[.*?\]\(\s*<?(.*?)\s*>?\)"
    image_matches = re.findall(pattern, content)
    
    # Use session cache to avoid redundant downloads from sandbox
    cache = cl.user_session.get("image_cache", {})
    
    elements = []
    seen_paths = set()
    displayed_hashes = cl.user_session.get("displayed_image_hashes", set())

    for img_path in image_matches:
        img_path = img_path.strip()
        if not img_path or img_path.startswith("http") or img_path in seen_paths:
            continue
            
        try:
            if img_path in cache:
                img_bytes = cache[img_path]
            else:
                logger.info(f"Loading image from sandbox for markdown: {img_path}")
                img_bytes = await sandbox.files.read(img_path, format="bytes")
                cache[img_path] = img_bytes
                cl.user_session.set("image_cache", cache)
            
            # Deduplicate by content hash
            img_hash = hashlib.md5(img_bytes).hexdigest()
            
            # If we've already shown this image hash in this TURN (as a standalone plot or in another message)
            # we set display="hidden". This allows markdown references to work without 
            # showing the image again at the bottom of the message.
            is_duplicate = img_hash in displayed_hashes
            
            displayed_hashes.add(img_hash)
            cl.user_session.set("displayed_image_hashes", displayed_hashes)

            elements.append(cl.Image(
                content=img_bytes, 
                name=img_path, 
                display="hidden" if is_duplicate else "inline"
            ))
            seen_paths.add(img_path)
        except Exception as e:
            logger.warning(f"Failed to load markdown image {img_path}: {e}")
            
    return elements

@cl.on_message
async def main(message: cl.Message):
    """
    Process incoming messages and run the agent graph.
    """
    logger.info(f"Received message: {message.content[:50]}...")
    state = cl.user_session.get("state")
    sandbox = cl.user_session.get("sandbox")

    if not state or not sandbox:
        await cl.ErrorMessage(content="نشست (Session) به درستی راه‌اندازی نشده است.").send()
        return

    # Reset displayed image hashes and cache for the new turn
    cl.user_session.set("displayed_image_hashes", set())
    cl.user_session.set("image_cache", {})

    # 1. Handle File Uploads
    if message.elements:
        for element in message.elements:
            if isinstance(element, cl.File):
                filename = element.name
                logger.info(f"User is uploading file: {filename}")
                
                await cl.Message(content=f"در حال آپلود `{filename}` به محیط مجازی...").send()
                
                # Write to sandbox - read from path as content might be None in some versions
                if element.path and os.path.exists(element.path):
                    with open(element.path, "rb") as f:
                        content = f.read()
                    await sandbox.files.write(filename, content)
                elif element.content:
                    await sandbox.files.write(filename, element.content)
                else:
                    await cl.ErrorMessage(content=f"عدم امکان خواندن محتوای فایل `{filename}`").send()
                    continue
                
                # Notify state
                state["messages"].append(HumanMessage(content=f"[System: User uploaded file '{filename}']"))
                await cl.Message(content=f"فایل `{filename}` با موفقیت به مسیر `{state['cwd']}` آپلود شد.").send()

    # 2. Process User Prompt
    state["messages"].append(HumanMessage(content=message.content))
    
    config = {
        "recursion_limit": 1000,
        "configurable": {"sandbox": sandbox}
    }

    # 3. Execute Graph and Stream results
    active_steps = {} # To track cl.Step/Message instances by node name
    last_worker_node = None # To track which node last called a tool

    logger.info("Starting graph execution...")
    try:
        async for event in graph.astream(state, config=config):
            for node_name, value in event.items():
                # Create a UI object for the node if it doesn't exist
                if node_name not in active_steps and node_name != Nodes.TOOLS:
                    if node_name == Nodes.SUPERVISOR:
                        ui_obj = cl.Step(name=node_name,parent_id=cl.context.current_step.id)
                    else:
                        ui_obj = cl.Message(content="", author=node_name)
                    active_steps[node_name] = ui_obj
                    await ui_obj.send()
                else:
                    ui_obj = active_steps.get(node_name)

                # Update UI content or send sub-messages
                if node_name in [Nodes.CLEANER, Nodes.EDA, Nodes.SUPERVISOR, Nodes.TRAINER, Nodes.STORYTELLER, Nodes.REPORTER, Nodes.FEATURE_ENGINEER]:
                    last_worker_node = node_name
                    if "messages" in value:
                        last_msg = value["messages"][-1]
                        state["messages"].append(last_msg)
                        
                        if last_msg.content:
                            # Accumulate messages
                            if isinstance(ui_obj, cl.Step):
                                current_output = ui_obj.output if ui_obj.output else ""
                                ui_obj.output = (current_output + "\n\n" + last_msg.content).strip()
                                # Scan for and attach images
                                ui_obj.elements = await get_images_from_markdown(ui_obj.output, sandbox)
                                await ui_obj.update()
                            else:
                                current_content = ui_obj.content if ui_obj.content else ""
                                ui_obj.content = (current_content + "\n\n" + last_msg.content).strip()
                                # Scan for and attach images
                                ui_obj.elements = await get_images_from_markdown(ui_obj.content, sandbox)
                                await ui_obj.update()

                        # Handle Tool Calls
                        if hasattr(last_msg, 'tool_calls') and last_msg.tool_calls:
                            for tc in last_msg.tool_calls:
                                tool_name = tc['name']
                                
                                # Content formatting
                                if tool_name == "run_python" and "code" in tc['args']:
                                    tool_content = f"```python\n{tc['args']['code']}\n```"
                                    image_elements = []
                                elif tool_name == "run_shell" and "command" in tc['args']:
                                    tool_content = f"```bash\n{tc['args']['command']}\n```"
                                    image_elements = []
                                elif tool_name == "create_markdown" and "content" in tc['args']:
                                    tool_content = tc['args']['content']
                                    # Scan tool arguments for images
                                    image_elements = await get_images_from_markdown(tool_content, sandbox)
                                else:
                                    tool_content = f"```json\n{str(tc['args'])}\n```"
                                    image_elements = []

                                # UI Display: Step for Supervisor, Message for others
                                if node_name == Nodes.SUPERVISOR:
                                    tool_step = cl.Step(name=f"Calling: {tool_name}", parent_id=ui_obj.id)
                                    tool_step.output = tool_content
                                    if image_elements:
                                        tool_step.elements = image_elements
                                    await tool_step.send()
                                else:
                                    await cl.Message(content=tool_content, author=f"{node_name} (Tool)", elements=image_elements).send()

                    if "next" in value and value["next"] != Nodes.FINISH:
                        logger.debug(f"Routing to {value['next']}")

                elif node_name == Nodes.TOOLS:
                    state["messages"].extend(value["messages"])
                    state["notebook_cells"].extend(value["notebook_cells"])
                    
                    # Display tool results
                    for msg in value["messages"]:
                        # Skip markdown and download success messages
                        if msg.name in ["create_markdown", "download_file"]:
                            continue

                        # Format content
                        display_content = msg.content
                        if len(display_content) > 3000:
                            display_content = display_content[:3000] + "\n\n... (output truncated) ..."
                        
                        if msg.name in ["run_python", "run_shell"]:
                            formatted_content = f"```python\n{display_content}\n```"
                        else:
                            formatted_content = display_content

                        # UI Display: Nested Step for Supervisor, Simple Message for others
                        parent_ui = active_steps.get(last_worker_node)
                        if last_worker_node == Nodes.SUPERVISOR and parent_ui:
                            tool_res_step = cl.Step(name=f"Result: {msg.name}", parent_id=parent_ui.id)
                            tool_res_step.output = formatted_content
                            await tool_res_step.send()
                        else:
                            await cl.Message(content=formatted_content, author=f"{last_worker_node} (Result)").send()

                    # Check for and display images from Jupyter outputs
                    displayed_hashes = cl.user_session.get("displayed_image_hashes", set())

                    for cell in value.get("notebook_cells", []):
                        if cell.get("cell_type") == "code":
                            for output in cell.get("outputs", []):
                                if output.get("type") == "image":
                                    try:
                                        img_data = output.get("data")
                                        if isinstance(img_data, str):
                                            # Handle possible base64 padding or prefixes
                                            if "," in img_data:
                                                img_data = img_data.split(",")[1]
                                            img_bytes = base64.b64decode(img_data)
                                        else:
                                            img_bytes = img_data
                                        
                                        # Deduplicate by hash
                                        img_hash = hashlib.md5(img_bytes).hexdigest()
                                        if img_hash in displayed_hashes:
                                            logger.info("Skipping duplicate image (Jupyter output)")
                                            continue
                                        
                                        displayed_hashes.add(img_hash)
                                        cl.user_session.set("displayed_image_hashes", displayed_hashes)

                                        image = cl.Image(
                                            content=img_bytes, 
                                            name="plot.png", 
                                            display="inline"
                                        )
                                        await cl.Message(
                                            content="", 
                                            elements=[image], 
                                            author=f"{last_worker_node} (Plot)"
                                        ).send()
                                    except Exception as img_err:
                                        logger.error(f"Failed to display image: {img_err}")

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
            await cl.Message(content="### دانلود خروجی‌های نشست ###", elements=files_to_send).send()

        logger.info("Graph execution completed successfully.")

    except Exception as e:
        logger.error(f"Error during graph execution: {e}", exc_info=True)
        await cl.ErrorMessage(content=f"یک خطا رخ داد: {str(e)}").send()

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
            await cl.Message(content=f"نشست با موفقیت در فایل `{filename}` ذخیره شد.").send()
        except Exception as e:
            logger.error(f"Failed to export notebook: {e}")

    if sandbox:
        await sandbox.kill()
        logger.info("E2B Sandbox closed.")