import asyncio
import os
from langchain_core.messages import HumanMessage
from e2b_code_interpreter import AsyncSandbox

from ds_agent.core.agent import create_graph
from ds_agent.utils.notebook import save_session_to_ipynb
from ds_agent.config import settings
from ds_agent.utils.logger import logger

async def main():
    logger.info("Initializing Data Science Agent...")
    graph = create_graph()
    
    # Initial state
    state = {
        "messages": [],
        "notebook_cells": [],
        "sandbox_session": None,
        "cwd": "/home/user"
    }
    
    print("\nAgent ready. Type 'exit' or 'quit' to stop.")
    print("Commands: /upload <path> to upload a file.\n")
    
    try:
        # Use E2B AsyncSandbox as a context manager for guaranteed cleanup
        async with await AsyncSandbox.create(api_key=settings.e2b_api_key) as sandbox:
            state["sandbox_session"] = sandbox
            logger.info("E2B AsyncSandbox initialized and active.")

            while True:
                try:
                    user_input = input("User: ")
                    if user_input.lower() in ["exit", "quit"]:
                        break
                    
                    # Handle file uploads
                    if user_input.startswith("/upload "):
                        file_path = user_input.replace("/upload ", "").strip()
                        if not os.path.exists(file_path):
                            print(f"Error: File '{file_path}' not found.")
                            continue
                        
                        filename = os.path.basename(file_path)
                        print(f"Uploading {filename} to sandbox...")
                        with open(file_path, "rb") as f:
                            await sandbox.files.write(filename, f)
                        
                        upload_msg = f"Successfully uploaded {filename} to the current directory."
                        print(f"System: {upload_msg}")
                        state["messages"].append(HumanMessage(content=f"[System: User uploaded file '{filename}']"))
                        continue

                    # Process user message
                    state["messages"].append(HumanMessage(content=user_input))
                    
                    config = {"recursion_limit": 50} 
                    async for event in graph.astream(state, config=config):
                        for key, value in event.items():
                            if key == "agent":
                                last_msg = value["messages"][-1]
                                state["messages"].append(last_msg)
                                if last_msg.content:
                                    print(f"\nAgent: {last_msg.content}\n")
                            elif key == "tools":
                                state["messages"].extend(value["messages"])
                                state["notebook_cells"].extend(value["notebook_cells"])
                                # sandbox_session is already set via context manager
                                logger.info("Tool execution cycle completed")
                    
                except KeyboardInterrupt:
                    break
                except Exception as e:
                    logger.error(f"An error occurred during turn: {e}", exc_info=True)
                    break

    except Exception as e:
        logger.error(f"Failed to initialize sandbox or main loop crashed: {e}")
    finally:
        # Export logic
        if state["notebook_cells"]:
            logger.info("Exporting session to notebook...")
            try:
                filename = save_session_to_ipynb(state, "analysis.ipynb")
                print(f"\nNotebook exported to {filename}")
            except Exception as e:
                logger.error(f"Failed to save notebook: {e}")
        
        logger.info("Data Science Agent session finished.")

if __name__ == "__main__":
    asyncio.run(main())