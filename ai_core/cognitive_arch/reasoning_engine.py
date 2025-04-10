# Devin/ai_core/cognitive_arch/reasoning_engine.py

import time
from typing import Any, Dict, List, Optional, Tuple

# Import memory components (assuming they are in the same directory or path is set)
try:
    from .working_memory import WorkingMemory
    from .long_term_memory import LongTermMemory
except ImportError:
    # Handle potential import errors if run standalone or structure changes
    print("Warning: Could not import memory modules using relative paths.")
    WorkingMemory = object # Placeholder
    LongTermMemory = object # Placeholder

# Placeholder types for collaborators (these would be defined elsewhere)
class AIIntegrationClient: # Placeholder for interacting with external LLMs (ChatGPT, Gemini, etc.)
    def query(self, prompt: str, context: Optional[str] = None) -> str:
        print(f"Placeholder AI Client: Querying LLM with prompt (len={len(prompt)})...")
        # Simulate LLM response
        if "plan" in prompt.lower():
            return "Plan:\n1. Search LTM for 'example.com vulnerabilities'.\n2. If nothing found, use web search.\n3. Analyze results.\n4. Formulate report."
        elif "clarify" in prompt.lower():
            return "Clarification needed: Please specify the type of vulnerabilities (e.g., web, network)."
        else:
            return "Placeholder LLM response based on analysis."

class ActionDispatcher: # Placeholder for sending planned actions to execution modules (after permission)
    def dispatch(self, action_type: str, params: Dict[str, Any]) -> Any:
        print(f"Placeholder Action Dispatcher: Dispatching action '{action_type}' with params {params}...")
        # Simulate action result
        if action_type == "web_search":
            return f"Web search results for '{params.get('query')}'..."
        elif action_type == "run_tool":
            return f"Output from running tool '{params.get('tool_name')}'..."
        elif action_type == "ask_user":
            return f"Asking user: '{params.get('question')}'" # In reality, this would trigger UI interaction
        else:
            return f"Result of action '{action_type}'."

class ReasoningEngine:
    """
    The core reasoning component of the AI.

    Processes user requests or goals, interacts with memory systems,
    plans steps (potentially using Chain-of-Thought or similar methods),
    delegates actions, and manages the overall task execution flow.
    """

    def __init__(self,
                 working_memory: WorkingMemory,
                 long_term_memory: LongTermMemory,
                 ai_client: AIIntegrationClient, # Client for external LLMs
                 action_dispatcher: ActionDispatcher # Interface to execution modules
                 ):
        """
        Initializes the Reasoning Engine.

        Args:
            working_memory (WorkingMemory): Instance for short-term context.
            long_term_memory (LongTermMemory): Instance for persistent knowledge.
            ai_client (AIIntegrationClient): Client to interact with external foundation models.
            action_dispatcher (ActionDispatcher): Interface to dispatch actions for execution.
        """
        if not isinstance(working_memory, WorkingMemory):
             # Use placeholder check if imports failed
             if WorkingMemory is not object and not isinstance(working_memory, object):
                 raise TypeError("working_memory must be an instance of WorkingMemory")
        if not isinstance(long_term_memory, LongTermMemory):
             if LongTermMemory is not object and not isinstance(long_term_memory, object):
                raise TypeError("long_term_memory must be an instance of LongTermMemory")

        self.wm = working_memory
        self.ltm = long_term_memory
        self.ai_client = ai_client
        self.action_dispatcher = action_dispatcher
        print("ReasoningEngine initialized.")

    def process_request(self, user_request: str, max_steps: int = 10) -> Dict[str, Any]:
        """
        Processes a user request or goal through a reasoning loop.

        Args:
            user_request (str): The initial request or goal from the user.
            max_steps (int): Maximum number of reasoning steps to prevent infinite loops.

        Returns:
            Dict[str, Any]: A dictionary containing the final result, status, and possibly intermediate steps.
        """
        print(f"\n--- Reasoning Engine Processing Request ---")
        print(f"Initial Request: '{user_request}'")

        # Initialize state for this request
        self.wm.clear_memory() # Clear working memory for new request
        self.wm.add_item("initial_request", user_request)
        self.wm.add_item("current_goal", user_request) # Initial goal is the request itself
        step_count = 0
        final_result = None
        status = "Processing"

        while step_count < max_steps:
            step_count += 1
            print(f"\n[Step {step_count}/{max_steps}]")

            # 1. THINK: Analyze current state and decide next step
            current_state_summary = self._summarize_current_state()
            print(f"  Thinking based on state: {current_state_summary}")
            next_step_or_action_plan = self._think(current_state_summary) # Might involve LLM call

            # 2. FORMULATE ACTION: Decide *what specific action* to take based on the plan
            action_type, action_params = self._formulate_action(next_step_or_action_plan)
            print(f"  Formulated Action: Type='{action_type}', Params={action_params}")

            if action_type == "final_answer":
                print("  Action is to provide final answer.")
                final_result = action_params.get("answer", "No specific answer formulated.")
                status = "Completed"
                break # Exit loop

            if action_type == "clarification_needed":
                print("  Action is to ask user for clarification.")
                # In a real system, this would trigger interaction via HMI/API Gateway
                # Here, we simulate dispatching it.
                observation = self.action_dispatcher.dispatch(action_type, action_params)
                status = "Waiting for User Input"
                final_result = observation # Store the question asked
                break # Exit loop (waiting for external input)

            if action_type == "no_action":
                print("  No further action determined. Ending process.")
                status = "Stuck"
                final_result = "Could not determine next step."
                break # Exit loop


            # 3. ACT (Dispatch): Send the formulated action for execution
            #    *** CRITICAL: In a real system, the ActionDispatcher MUST interact
            #    *** with the User Permission System BEFORE executing potentially
            #    *** impactful actions (run_tool, file_io, web_control, etc.)
            print(f"  Dispatching Action: {action_type}")
            try:
                action_result = self.action_dispatcher.dispatch(action_type, action_params)
                observation = f"Action '{action_type}' executed. Result: {str(action_result)[:200]}..." # Limit result length
            except Exception as e:
                print(f"  Error dispatching action '{action_type}': {e}")
                observation = f"Error executing action '{action_type}': {e}"
                status = "Error"
                final_result = observation
                # Potentially add error details to working memory here
                # break # Decide if errors should halt the process

            # 4. OBSERVE: Update working memory with the result/observation
            print(f"  Observing result: {observation}")
            self.wm.add_item(f"observation_step_{step_count}", observation)
            self.wm.add_item("last_action_result", action_result) # Store potentially structured result

            # Optional: Update current goal if the plan involved sub-goals
            # self.wm.add_item("current_goal", new_sub_goal)

        # Loop finished or broken
        if status == "Processing":
            status = "Max Steps Reached"
            final_result = "Reached maximum reasoning steps without a final answer."
            print(f"Warning: Max steps ({max_steps}) reached.")

        print(f"--- Reasoning Engine Finished ---")
        print(f"Final Status: {status}")
        print(f"Final Result: {final_result}")

        return {"status": status, "result": final_result, "steps_taken": step_count}

    def _summarize_current_state(self) -> str:
        """ Creates a summary of the current state from working memory for reasoning. """
        # This should be more sophisticated, perhaps selecting key items.
        recent_items = self.wm.get_recent_items(count=5) # Get last 5 items/observations
        goal = self.wm.get_item("current_goal")
        summary = f"Current Goal: {goal}\nRecent Activity:\n"
        for ts, key, val in reversed(recent_items): # Show most recent first
            summary += f"- {key}: {str(val)[:100]}...\n" # Limit value length
        return summary.strip()

    def _think(self, current_state_summary: str) -> str:
        """
        Performs the core reasoning/planning step.
        This could involve complex logic or calling an external LLM.
        Returns a plan or description of the next step/action needed.
        """
        # --- Placeholder for complex reasoning ---
        # Option 1: Simple rule-based logic (very basic)
        # if "error" in current_state_summary.lower(): return "Action: Report error."
        # if "vulnerabilities found" in current_state_summary.lower(): return "Action: Formulate report."

        # Option 2: Call external LLM for planning/reasoning (more likely)
        prompt = f"""Given the current state:\n{current_state_summary}\n\n
        Based ONLY on the information provided, determine the single best next action or step to achieve the current goal.
        Possible actions include: query_ltm(query), web_search(query), run_tool(tool_name, args), formulate_report(), ask_user(question), final_answer(answer), no_action().
        Provide JUST the next action and its parameters, or a brief plan leading to the next action. If the goal is met, use final_answer. If more info is needed from the user, use ask_user.
        Example response: run_tool(tool_name='nmap', args='-sV example.com')
        Example response: ask_user(question='What specific domain should I scan?')
        Example response: final_answer(answer='Scan complete. Found 3 vulnerabilities.')
        """
        llm_suggestion = self.ai_client.query(prompt=prompt)
        # --- End Placeholder ---

        return llm_suggestion # Return the plan/action suggested by LLM or logic

    def _formulate_action(self, plan_or_action_string: str) -> Tuple[str, Dict[str, Any]]:
        """
        Parses the output of the _think step into a specific action type and parameters.
        This needs robust parsing based on the expected output format from _think.
        """
        # --- Placeholder for parsing logic ---
        # This needs to reliably parse strings like:
        # "run_tool(tool_name='nmap', args='-sV example.com')"
        # "ask_user(question='What specific domain?')"
        # "final_answer(answer='Report text...')"
        # "web_search(query='latest CVEs')"

        plan_lower = plan_or_action_string.lower()
        # Very basic parsing example:
        if plan_lower.startswith("run_tool"):
            # TODO: Implement robust parsing for args
            tool_name = plan_or_action_string[plan_or_action_string.find("tool_name='")+len("tool_name='"):plan_or_action_string.find("'", plan_or_action_string.find("tool_name='")+len("tool_name='"))]
            args = plan_or_action_string[plan_or_action_string.find("args='")+len("args='"):-1] # Simple arg extraction
            return "run_tool", {"tool_name": tool_name, "args": args}
        elif plan_lower.startswith("ask_user"):
             # TODO: Implement robust parsing for question
            question = plan_or_action_string[plan_or_action_string.find("question='")+len("question='"):-1]
            return "ask_user", {"question": question}
        elif plan_lower.startswith("final_answer"):
             # TODO: Implement robust parsing for answer
            answer = plan_or_action_string[plan_or_action_string.find("answer='")+len("answer='"):-1]
            return "final_answer", {"answer": answer}
        elif plan_lower.startswith("web_search"):
             # TODO: Implement robust parsing for query
            query = plan_or_action_string[plan_or_action_string.find("query='")+len("query='"):-1]
            return "web_search", {"query": query}
        elif plan_lower.startswith("query_ltm"):
             # TODO: Implement robust parsing for query
            query = plan_or_action_string[plan_or_action_string.find("query='")+len("query='"):-1]
            return "query_ltm", {"query": query} # Assuming dispatcher handles LTM query
        # Add more action types as needed (file_io, web_control, etc.)
        else:
            # If parsing fails or plan is unclear, maybe default to asking user or stopping
            print(f"Warning: Could not parse plan into specific action: '{plan_or_action_string}'")
            return "clarification_needed", {"question": f"I received an unclear plan: '{plan_or_action_string[:100]}...'. Could you please clarify the goal or next step?"}
        # --- End Placeholder ---


# Example Usage (conceptual)
if __name__ == "__main__":
    print("\n--- Reasoning Engine Example ---")

    # Create mock/placeholder instances for dependencies
    # Use actual classes if they exist and are importable
    mock_wm = WorkingMemory(max_size=10) # Assuming WorkingMemory class exists from previous step
    mock_ltm = LongTermMemory() # Assuming LongTermMemory class exists from previous step
    mock_ai_client = AIIntegrationClient()
    mock_dispatcher = ActionDispatcher()

    # Initialize the engine
    engine = ReasoningEngine(mock_wm, mock_ltm, mock_ai_client, mock_dispatcher)

    # Process a sample request
    request = "Perform a basic security scan on example.com"
    result_info = engine.process_request(request, max_steps=5) # Limit steps for example

    print("\n--- Final Result Info ---")
    print(f"Status: {result_info['status']}")
    print(f"Result: {result_info['result']}")
    print(f"Steps: {result_info['steps_taken']}")

    print("\n--- Example with potential clarification ---")
    request_clarify = "Update the system."
    result_info_clarify = engine.process_request(request_clarify, max_steps=3)

    print("\n--- Final Result Info (Clarification) ---")
    print(f"Status: {result_info_clarify['status']}")
    print(f"Result: {result_info_clarify['result']}")
    print(f"Steps: {result_info_clarify['steps_taken']}")


    print("--- End Example ---")
