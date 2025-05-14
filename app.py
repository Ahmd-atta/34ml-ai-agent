"""
Main CLI entry point for 34ML Social-Media AI Agent
LangGraph multi-agent orchestrator
"""

import logging
import builtins
from langgraph.checkpoint.memory import MemorySaver
from build_graph import get_runner

# Setup logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")

# Initialize checkpointer
checkpointer = MemorySaver()

# Get graph runner
runner = get_runner(checkpointer=checkpointer)

def main():
    print("=== 34ML Agent (type 'help' for scheduler commands, 'quit' to exit) ===")
    
    while True:
        user_input = input("You: ").strip()
        logging.info(f"Processing input: {user_input}")
        
        if user_input.lower() == "quit":
            logging.info("Exiting CLI")
            # Ensure final state is checkpointed
            try:
                current_state = runner.get_state({"configurable": {"thread_id": "default"}})
                if current_state:
                    logging.debug(f"Final state before exit: {current_state.values.get('conversation_history', [])}")
            except Exception as e:
                logging.error(f"Error retrieving state on exit: {e}")
            break
        
        if user_input.lower() == "help":
            print("""Scheduler commands:
  show queue | show <channel> queue
  show posts | show <channel> posts
  show scheduled posts | show scheduled <channel> posts
  show history
  schedule last [<channel>] post for <date>
  schedule <id> for <date>
  remove last [<channel>] | remove <id> [from <date>]
""")
            continue

        # Store raw input for generator
        builtins._last_user_raw = user_input

        # Process input through LangGraph
        try:
            # Use a consistent thread_id for persistence
            result = runner.invoke(
                {"user_input": user_input, "generated": False},
                config={"configurable": {"thread_id": "default"}}
            )
            bot_response = result.get("result", "No result returned. Try another command.")
            print(f"Bot: {bot_response}")
            # Update conversation history with bot response
            history = result.get("conversation_history", [])
            if history and history[-1]["user"] == user_input and not history[-1]["bot"]:
                history[-1]["bot"] = bot_response
            # Debug state persistence
            logging.debug(f"State after invoke: {history}")
        except Exception as e:
            logging.error(f"Error processing input: {e}")
            print(f"Bot: Error: {e}")

if __name__ == "__main__":
    main()