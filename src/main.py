import warnings
warnings.filterwarnings("ignore")

import asyncio
import os
from langchain_core.messages import HumanMessage
from e2b_code_interpreter import AsyncSandbox

from ds_agent.core.graph import create_graph
from ds_agent.utils.notebook import save_session_to_ipynb
from ds_agent.config import settings, Nodes
from ds_agent.utils.logger import logger

async def main():
    logger.info("Initializing Data Science Agent...")
    graph = create_graph()
    
    # Initial state
    state = {
        "messages": [],
        "notebook_cells": [],
        "cwd": "/home/user",
        "next": Nodes.SUPERVISOR,
        "node_visits": {}
    }
    
    print("\nAgent ready. Type 'exit' or 'quit' to stop.")
    print("Commands: /upload <path> to upload a file.\n")
    
    try:
        # Use E2B AsyncSandbox as a context manager for guaranteed cleanup
        async with await AsyncSandbox.create(
            api_key=settings.e2b_api_key.get_secret_value(),
            timeout=settings.sandbox_timeout
        ) as sandbox:
            logger.info(f"E2B AsyncSandbox initialized and active (Timeout: {settings.sandbox_timeout}s).")

            while True:
                try:
                    # 1. Get optional file path
                    file_input = input("File path to upload (optional, press Enter to skip): ").strip()
                    if file_input.lower() in ["exit", "quit"]:
                        break
                    
                    if file_input:
                        if not os.path.exists(file_input):
                            print(f"Error: Local file '{file_input}' not found.")
                        else:
                            filename = os.path.basename(file_input)
                            print(f"Uploading {filename} to sandbox...")
                            with open(file_input, "rb") as f:
                                await sandbox.files.write(filename, f)
                            print(f"System: Successfully uploaded {filename}.")
                            state["messages"].append(HumanMessage(content=f"[System: User uploaded file '{filename}']"))

                    # 2. Get user prompt
                    user_input = input("User prompt: ").strip()
                    if not user_input:
                        continue
                    if user_input.lower() in ["exit", "quit"]:
                        break
                    
                    # Process user message
                    state["messages"].append(HumanMessage(content=user_input))
                    
                    config = {
                        "recursion_limit": 1000,
                        "configurable": {"sandbox": sandbox}
                    }
                    
                    async for event in graph.astream(state, config=config):
                        for key, value in event.items():
                            # Handle worker, supervisor and reporter nodes
                            if key in [Nodes.CLEANER, Nodes.EDA, Nodes.SUPERVISOR, Nodes.REPORTER]:
                                if "messages" in value:
                                    last_msg = value["messages"][-1]
                                    state["messages"].append(last_msg)
                                    
                                    # Show thinking if there's content
                                    if last_msg.content:
                                        print(f"\n--- Agent Thinking ({key}) ---\n{last_msg.content}\n")
                                    
                                    # Show tool calls if any
                                    if hasattr(last_msg, 'tool_calls') and last_msg.tool_calls:
                                        print("--- Tool Calls Requested ---")
                                        for tc in last_msg.tool_calls:
                                            print(f"Tool: {tc['name']}")
                                            print(f"Arguments: {tc['args']}")
                                        print("---------------------------\n")
                                
                                if "next" in value:
                                    state["next"] = value["next"]
                                    if value["next"] != Nodes.FINISH:
                                        print(f"--- Supervisor Routing: Next is {value['next']} ---")

                            elif key == Nodes.TOOLS:
                                logger.info("Tool execution cycle completed")
                                # Update local state
                                state["messages"].extend(value["messages"])
                                state["notebook_cells"].extend(value["notebook_cells"])
                                
                                print("--- Tool Execution Results ---")
                                for msg in value["messages"]:
                                    # Truncate long outputs for display
                                    content = msg.content
                                    if len(content) > 500:
                                        content = content[:500] + "..."
                                    print(f"Tool '{msg.name}' result:\n{content}\n")
                                print("------------------------------\n")
                    
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