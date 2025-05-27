# Devin/modules/automation_module.py
# Purpose: Defines and executes user-defined or system-learned workflows and routines.
#          Manages sequences of actions that Devin can perform.
# Automates user workflows and routines ⚙️📜

import logging
import uuid
from datetime import datetime, timezone
from enum import Enum, auto
from typing import List, Dict, Any, Optional, Callable, Union
from dataclasses import dataclass, field

# Configure basic logging
logger = logging.getLogger("AutomationModule")
if not logger.handlers: # Prevent duplicate handlers
    _console_handler = logging.StreamHandler()
    _console_handler.setFormatter(logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s'))
    logger.addHandler(_console_handler)
    logger.setLevel(logging.INFO)

class ActionStatus(Enum):
    PENDING = auto()
    RUNNING = auto()
    COMPLETED = auto()
    FAILED = auto()
    SKIPPED = auto()

@dataclass
class WorkflowStep:
    """Represents a single step in an automated workflow."""
    step_id: str = field(default_factory=lambda: f"step_{uuid.uuid4().hex[:8]}")
    description: Optional[str] = None
    action_type: str # e.g., "BROWSE_URL", "EXECUTE_CODE", "SEND_MESSAGE", "PC_MOUSE_CLICK", "PC_KEYBOARD_TYPE"
    parameters: Dict[str, Any] = field(default_factory=dict) # Parameters for the action
    # For conceptual ordering, not a full dependency graph in this simple version
    depends_on_step_id: Optional[str] = None 
    # To store output of this step that might be used by subsequent steps
    output_variable_name: Optional[str] = None 

@dataclass
class Workflow:
    """Represents a defined workflow consisting of multiple steps."""
    workflow_id: str = field(default_factory=lambda: f"wf_{uuid.uuid4().hex[:8]}")
    name: str
    description: Optional[str] = None
    steps: List[WorkflowStep] = field(default_factory=list)
    creation_date: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    version: str = "1.0"

@dataclass
class WorkflowExecutionLog:
    """Logs the execution of a workflow."""
    workflow_id: str
    execution_id: str = field(default_factory=lambda: f"exec_{uuid.uuid4().hex[:8]}")
    start_time: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    end_time: Optional[datetime] = None
    status: ActionStatus = ActionStatus.PENDING
    step_results: List[Dict[str, Any]] = field(default_factory=list) # {"step_id", "status", "result", "error"}
    final_context: Dict[str, Any] = field(default_factory=dict) # Context after execution, including outputs

class ConceptualDevinCapabilities:
    """
    A placeholder class representing the various capabilities of Devin
    that can be called by the AutomationModule.
    In a real system, these would trigger other modules or actions.
    """
    def __init__(self):
        logger.info("ConceptualDevinCapabilities interface initialized.")

    def _log_action(self, action_name: str, params: Dict) -> Dict[str, Any]:
        logger.info(f"DEVIN_CAPABILITY: '{action_name}' called with params: {params}")
        # Simulate some generic success or data return
        if "return_value" in params:
            return {"status": "success", "message": f"'{action_name}' executed.", "data": params["return_value"]}
        return {"status": "success", "message": f"'{action_name}' executed."}

    def browse_url(self, url: str, timeout_seconds: int = 30) -> Dict[str, Any]:
        return self._log_action("browse_url", {"url": url, "timeout_seconds": timeout_seconds, "return_value": f"Page content from {url} (simulated)"})

    def execute_python_code(self, code_block: str, environment: str = "sandbox") -> Dict[str, Any]:
        return self._log_action("execute_python_code", {"code_block": code_block, "environment": environment, "return_value": "Code execution result (simulated)"})

    def send_message_to_user(self, user_id: str, message: str) -> Dict[str, Any]:
        # This might internally use the AIConversationModule
        return self._log_action("send_message_to_user", {"user_id": user_id, "message": message})

    def pc_mouse_click(self, x: int, y: int, button: str = "left", clicks: int = 1) -> Dict[str, Any]:
        return self._log_action("pc_mouse_click", {"x": x, "y": y, "button": button, "clicks": clicks})
    
    def pc_keyboard_type(self, text_to_type: str, interval_seconds: float = 0.01) -> Dict[str, Any]:
        return self._log_action("pc_keyboard_type", {"text_to_type": text_to_type, "interval_seconds": interval_seconds})

    def get_user_input(self, prompt_message: str, user_id: str) -> Dict[str, Any]:
        # This might use AIConversationModule or a direct UI prompt
        simulated_input = f"User input for '{prompt_message}' (simulated)"
        return self._log_action("get_user_input", {"prompt_message": prompt_message, "user_id": user_id, "return_value": simulated_input})
    
    def run_shell_command(self, command: str, working_directory: Optional[str] = None) -> Dict[str, Any]:
        return self._log_action("run_shell_command", {"command": command, "working_directory": working_directory, "return_value": f"Output of '{command}' (simulated)"})


class AutomationEngine:
    """
    Manages the definition and execution of automated workflows.
    """
    def __init__(self, devin_capabilities: ConceptualDevinCapabilities):
        self.devin_caps = devin_capabilities
        self.workflows: Dict[str, Workflow] = {}
        self.execution_logs: Dict[str, WorkflowExecutionLog] = {}
        logger.info("AutomationEngine initialized.")

    def register_workflow(self, workflow: Workflow) -> None:
        """Registers a new workflow or updates an existing one."""
        if not isinstance(workflow, Workflow):
            raise TypeError("Invalid workflow object provided.")
        self.workflows[workflow.workflow_id] = workflow
        logger.info(f"Workflow '{workflow.name}' (ID: {workflow.workflow_id}) registered/updated.")

    def _resolve_parameters(self, params_template: Dict[str, Any], context: Dict[str, Any]) -> Dict[str, Any]:
        """
        Resolves parameters, potentially substituting values from the context.
        Example: If a param value is "$step1_output.data_field", it tries to get it from context.
        This is a very simplified version.
        """
        resolved_params = {}
        for key, value in params_template.items():
            if isinstance(value, str) and value.startswith("$"):
                # Simple context variable substitution: $variable_name or $step_output_var.key
                var_path = value[1:].split('.')
                current_val = context
                try:
                    for path_part in var_path:
                        if isinstance(current_val, dict):
                            current_val = current_val.get(path_part)
                        else: # Cannot traverse further
                            current_val = None
                            break
                    resolved_params[key] = current_val if current_val is not None else value # Fallback to original string if not found
                except Exception:
                    resolved_params[key] = value # Fallback
                    logger.warning(f"Could not resolve context variable '{value}' for parameter '{key}'. Using as literal.")
            else:
                resolved_params[key] = value
        return resolved_params

    def execute_workflow(self, workflow_id: str, initial_context: Optional[Dict[str, Any]] = None) -> WorkflowExecutionLog:
        """
        Executes a registered workflow.

        Args:
            workflow_id (str): The ID of the workflow to execute.
            initial_context (Optional[Dict[str, Any]]): Initial data/context for the workflow.

        Returns:
            WorkflowExecutionLog: A log detailing the execution.
        """
        workflow = self.workflows.get(workflow_id)
        if not workflow:
            logger.error(f"Workflow with ID '{workflow_id}' not found.")
            # Return a failed execution log
            exec_log = WorkflowExecutionLog(workflow_id=workflow_id, status=ActionStatus.FAILED)
            exec_log.step_results.append({"step_id": "N/A", "status": ActionStatus.FAILED, "error": "Workflow not found."})
            exec_log.end_time = datetime.now(timezone.utc)
            return exec_log

        logger.info(f"Starting execution of workflow '{workflow.name}' (ID: {workflow_id}).")
        execution_log = WorkflowExecutionLog(workflow_id=workflow_id, status=ActionStatus.RUNNING)
        
        # Context stores outputs of previous steps, identified by `output_variable_name`
        current_context = initial_context.copy() if initial_context else {}
        
        # Super simplified step ordering: assumes linear or correctly ordered steps for now
        # A real system would handle `depends_on_step_id` with a graph or topological sort.
        for step in workflow.steps:
            step_result_log = {"step_id": step.step_id, "description": step.description, "action": step.action_type}
            logger.info(f"  Executing step '{step.step_id}': {step.description or step.action_type}")

            try:
                # Resolve parameters using current context
                resolved_params = self._resolve_parameters(step.parameters, current_context)
                
                action_method_name = step.action_type.lower() # Assuming action_type maps to method names
                if hasattr(self.devin_caps, action_method_name):
                    action_func = getattr(self.devin_caps, action_method_name)
                    # Call the capability method with resolved parameters
                    result = action_func(**resolved_params)
                    
                    step_result_log["status"] = ActionStatus.COMPLETED
                    step_result_log["result"] = result
                    logger.info(f"    Step '{step.step_id}' completed. Result: {str(result)[:100]}...")
                    
                    if step.output_variable_name:
                        current_context[step.output_variable_name] = result.get("data") # Store relevant data from result
                        logger.debug(f"    Stored output as '{step.output_variable_name}': {str(result.get('data'))[:100]}...")
                else:
                    raise NotImplementedError(f"Action type '{step.action_type}' (method '{action_method_name}') not implemented in DevinCapabilities.")

            except Exception as e:
                logger.error(f"    Error executing step '{step.step_id}': {e}")
                step_result_log["status"] = ActionStatus.FAILED
                step_result_log["error"] = str(e)
                execution_log.status = ActionStatus.FAILED # Mark whole workflow as failed
                execution_log.step_results.append(step_result_log)
                break # Stop workflow on first error for this simple implementation
            
            execution_log.step_results.append(step_result_log)

        if execution_log.status != ActionStatus.FAILED:
            execution_log.status = ActionStatus.COMPLETED
        
        execution_log.end_time = datetime.now(timezone.utc)
        execution_log.final_context = current_context
        self.execution_logs[execution_log.execution_id] = execution_log
        
        logger.info(f"Workflow '{workflow.name}' execution finished with status: {execution_log.status.name}.")
        return execution_log

    def get_workflow_execution_log(self, execution_id: str) -> Optional[WorkflowExecutionLog]:
        return self.execution_logs.get(execution_id)


# Example Usage
if __name__ == "__main__":
    print("=========================================================")
    print("=== Automation Module Prototype ⚙️📜 ===")
    print("=========================================================")

    # 1. Initialize Devin's conceptual capabilities and the Automation Engine
    devin_interface = ConceptualDevinCapabilities()
    automation_engine = AutomationEngine(devin_capabilities=devin_interface)

    # 2. Define a sample workflow
    sample_workflow = Workflow(
        name="Daily News Briefing and Summary",
        description="Fetches news from a URL, asks user for a keyword, then summarizes (conceptually).",
        steps=[
            WorkflowStep(
                description="Browse to a news website.",
                action_type="browse_url", # Maps to devin_interface.browse_url()
                parameters={"url": "https://simulated-news.com/today"},
                output_variable_name="news_page_content"
            ),
            WorkflowStep(
                description="Ask user for a keyword to focus on.",
                action_type="get_user_input",
                parameters={"prompt_message": "Enter a keyword for news summary:", "user_id": "automation_user"},
                output_variable_name="user_keyword",
                depends_on_step_id=None # For simplicity, assume linear for now
            ),
            WorkflowStep(
                description="Conceptually summarize news based on keyword (simulated by sending a message).",
                action_type="send_message_to_user",
                parameters={
                    "user_id": "automation_user",
                    # Example of using output from previous step (very basic substitution)
                    "message": "Summary for '$user_keyword' from news (simulated): Devin found relevant articles on '$news_page_content'[0:50]..."
                },
                depends_on_step_id=None
            ),
            WorkflowStep(
                description="Simulate typing a thank you note.",
                action_type="pc_keyboard_type",
                parameters={"text_to_type": "Automation complete. Thank you!"}
            )
        ]
    )
    automation_engine.register_workflow(sample_workflow)
    print(f"\nWorkflow '{sample_workflow.name}' registered with ID: {sample_workflow.workflow_id}")

    # 3. Execute the workflow
    print(f"\n--- Executing workflow: {sample_workflow.name} ---")
    execution_summary = automation_engine.execute_workflow(sample_workflow.workflow_id)

    # 4. Display execution log summary
    print("\n--- Workflow Execution Summary ---")
    print(f"Execution ID: {execution_summary.execution_id}")
    print(f"Workflow ID: {execution_summary.workflow_id}")
    print(f"Status: {execution_summary.status.name}")
    print(f"Started: {execution_summary.start_time}, Ended: {execution_summary.end_time}")
    
    print("\nStep Results:")
    for step_res in execution_summary.step_results:
        print(f"  Step ID: {step_res['step_id']} ({step_res.get('description', 'N/A')})")
        print(f"    Action: {step_res['action']}")
        print(f"    Status: {step_res['status'].name}")
        if step_res['status'] == ActionStatus.FAILED:
            print(f"    Error: {step_res['error']}")
        else:
            print(f"    Result: {str(step_res.get('result', 'N/A'))[:150]}...")
            
    print("\nFinal Context (includes step outputs):")
    for key, val in execution_summary.final_context.items():
        print(f"  {key}: {str(val)[:100]}...")


    # Example of a workflow that might fail (if action_type doesn't exist)
    failing_workflow = Workflow(
        name="Test Failing Workflow",
        steps=[WorkflowStep(action_type="NON_EXISTENT_ACTION", parameters={})]
    )
    automation_engine.register_workflow(failing_workflow)
    print(f"\n--- Executing failing workflow: {failing_workflow.name} ---")
    failing_exec_summary = automation_engine.execute_workflow(failing_workflow.workflow_id)
    print(f"Failing Workflow Status: {failing_exec_summary.status.name}")
    print(f"  Error in step: {failing_exec_summary.step_results[0].get('error')}")


    print("\n=========================================================")
    print("=== Automation Module Prototype Complete ===")
    print("=========================================================")
