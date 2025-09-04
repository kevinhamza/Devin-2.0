# ==============================================================================
#  DEVIN AGI - MAIN ENTRY POINT
# ==============================================================================
#  Author: Kevin Devin
#  Version: 1.0.0
#  License: MIT
#  Description: The core entry point for the Devin AGI. This script
#               initializes all components, starts background servers, and
#               runs the main operational loop.
# ==============================================================================

import logging
import threading
import os
import json
from typing import Dict, Any, List

# --- Load Environment Variables ---
from dotenv import load_dotenv
load_dotenv()

# --- Grand Integration: Import all major components of the Devin project ---

# Servers (to be run in the background)
from servers.cloud_integration_server import CloudIntegrationServer
from servers.analytics_server import AnalyticsServer
from servers.mobile_integration_server import MobileIntegrationServer
from servers.ai_learning_server import AILearningServer

# Core AI and Decision-Making Modules
from modules.all_ais_modules import AIAgent, AIProvider
from singularity.goal_system.utility_function import UtilityFunction, TaskCompletionComponent, SafetyComponent, EfficiencyComponent, ConstraintAdherenceComponent, Plan
from singularity.goal_system.ethics_constraints import ConstraintSet, DoNoHarmConstraint, DataPrivacyConstraint, HonestyAndTransparencyConstraint

# Core Tool Execution and Management
from modules.tool_executor import ToolExecutor
from modules.user_interaction_module import UserInteractionManager
from security.security_dashboard import SecurityDashboard

# Tool-Providing Facades and Managers
from modules.cloud_services_manager import CloudServicesManager
from modules.cloud_integration_module import CloudFacade
from modules.pentesting_tools.pentesting_facade import PentestingFacade
from modules.automation_tools import DesktopAutomator, WebAutomator
from modules.system_monitor import SystemMonitorFacade, LocalMonitor, RemoteMonitor
from modules.mobile_integration_module import MobileFacade
from modules.os_operations.universal_operations import UniversalOSOperator
from modules.knowledge_retrieval.code_retriever import CodeRetriever
from modules.data_logger import DataLogger

# Configure basic logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger("DevinAGI")


class DevinAGI:
    """
    The main class that orchestrates the entire AGI system.
    """
    def __init__(self):
        logger.info("Initializing Devin AGI...")
        self.is_running = True
        self.conversation_history: List[Dict] = []

        # --- 1. Start All Background Servers ---
        self.servers = self._start_background_servers()

        # --- 2. Initialize Core Brain Components ---
        logger.info("Initializing core cognitive modules...")
        self.agent = AIAgent(
            openai_api_key=os.getenv("OPENAI_API_KEY"),
            gemini_api_key=os.getenv("GEMINI_API_KEY"),
            perplexity_api_key=os.getenv("PERPLEXITY_API_KEY")
        )
        self.uim = UserInteractionManager()
        self.security_dashboard = SecurityDashboard()

        # --- 3. Initialize Value Alignment System ---
        logger.info("Initializing value alignment and ethical framework...")
        self.ethical_constraints = ConstraintSet(constraints=[
            DoNoHarmConstraint(),
            DataPrivacyConstraint(),
            HonestyAndTransparencyConstraint(ai_agent=self.agent)
        ])
        utility_components = {
            "TaskCompletion": TaskCompletionComponent(weight=0.5, ai_agent=self.agent),
            "Safety": SafetyComponent(weight=1.0, security_dashboard=self.security_dashboard),
            "Efficiency": EfficiencyComponent(weight=0.1),
            "ConstraintAdherence": ConstraintAdherenceComponent(weight=1.0),
        }
        self.utility_function = UtilityFunction(components=utility_components)

        # --- 4. Initialize All Tools and Facades ---
        logger.info("Initializing toolset and capabilities...")
        # Note: In a real deployment, credentials for tools would be managed securely.
        cloud_facade = CloudFacade() # Assumes credentials are in environment
        self.cloud_manager = CloudServicesManager(cloud_facade=cloud_facade, uim=self.uim)
        self.pentesting_facade = PentestingFacade()
        self.desktop_automator = DesktopAutomator()
        self.web_automator = WebAutomator()
        self.system_monitor = SystemMonitorFacade(monitors=[LocalMonitor()])
        self.mobile_facade = MobileFacade()
        self.os_operator = UniversalOSOperator()
        self.code_retriever = CodeRetriever(project_root=".")
        self.data_logger = DataLogger()

        # --- 5. Initialize the Master Tool Executor ---
        logger.info("Registering all tools with the Tool Executor...")
        self.tool_executor = ToolExecutor(
            ai_agent=self.agent,
            security_dashboard=self.security_dashboard,
            cloud_services_manager=self.cloud_manager,
            pentesting_facade=self.pentesting_facade,
            desktop_automator=self.desktop_automator,
            web_automator=self.web_automator,
            system_monitor_facade=self.system_monitor,
            mobile_facade=self.mobile_facade,
            os_operator=self.os_operator,
            code_retriever=self.code_retriever,
            data_logger=self.data_logger
        )

        logger.info("✅ Devin AGI Initialization Complete. Ready for instructions.")

    def _start_background_servers(self) -> Dict[str, threading.Thread]:
        """Initializes and starts all backend servers in daemon threads."""
        servers_to_start = {
            "CloudIntegration": {"class": CloudIntegrationServer, "port": 5002},
            "Analytics": {"class": AnalyticsServer, "port": 5004},
            "MobileIntegration": {"class": MobileIntegrationServer, "port": 5006},
            "AILearning": {"class": AILearningServer, "port": 5007},
        }
        threads = {}
        for name, config in servers_to_start.items():
            server_instance = config["class"]()
            thread = threading.Thread(
                target=server_instance.run,
                args=("127.0.0.1", config["port"]),
                daemon=True,
                name=f"{name}ServerThread"
            )
            thread.start()
            threads[name] = thread
            logger.info(f"{name} Server started in background on port {config['port']}.")
        return threads

    def run(self):
        """The main operational loop of the AGI."""
        goal = self.uim.get_user_input("\nPlease state your high-level goal: ")
        self.conversation_history.append({"role": "user", "content": f"My goal is: {goal}"})

        while self.is_running:
            # 1. THINK: Agent decides the next step
            tool_call = self.agent.get_tool_selection_response(
                self.conversation_history,
                self.tool_executor.get_available_tools()
            )

            if not tool_call or not isinstance(tool_call, dict) or "tool" not in tool_call:
                logger.info("Agent decided no further action is needed or failed to select a tool. Concluding task.")
                break

            # 2. VERIFY & CONSENT: Check the plan against ethical and utility functions
            plan = Plan(steps=[tool_call])
            violations = self.ethical_constraints.check_plan(plan)
            if any(v.severity == "VETO" for v in violations):
                logger.error(f"Ethical VETO: Plan '{tool_call}' violates a core principle. Aborting.")
                break
                
            utility_score = self.utility_function.evaluate_plan(plan, goal, [])
            if utility_score.total_utility < 0.1: # Arbitrary threshold for "good enough"
                 logger.warning(f"Plan '{tool_call}' has low utility ({utility_score.total_utility:.2f}). Asking AI to reconsider.")
                 self.conversation_history.append({"role": "system", "content": "That plan seems suboptimal or unsafe. Please propose a different approach."})
                 continue
            
            if self.tool_executor.is_dangerous(tool_call['tool']):
                if not self.uim.ask_for_confirmation(f"The next action is potentially dangerous: {tool_call}. Do you want to proceed?"):
                    logger.warning("Action aborted by user consent.")
                    self.conversation_history.append({"role": "system", "content": f"User denied permission to execute {tool_call['tool']}."})
                    continue

            # 3. ACT: Execute the approved plan
            result = self.tool_executor.execute_tool(tool_call)
            
            # 4. PERCEIVE & UPDATE: Add the action and result to history
            self.conversation_history.append({"role": "assistant", "content": json.dumps(tool_call)})
            self.conversation_history.append({"role": "tool", "content": json.dumps(result)})

            # Check for completion (simplified)
            if "complete" in str(result).lower() or "finished" in str(result).lower():
                 logger.info("Agent reported task completion.")
                 break
            
            if len(self.conversation_history) > 20: # Safety break
                logger.warning("Conversation limit reached. Ending session.")
                break

    def shutdown(self):
        """Gracefully shuts down all components."""
        logger.info("Shutting down Devin AGI...")
        self.is_running = False
        # The server threads are daemons, so they will exit when the main thread does.
        # In a more robust system, we would send shutdown requests to each server.
        logger.info("Shutdown complete.")


if __name__ == "__main__":
    # --- ASCII Art Banner ---
    print(r"""
    ██████╗ ███████╗██╗   ██╗██╗███╗   ██╗
    ██╔══██╗██╔════╝██║   ██║██║████╗  ██║
    ██║  ██║█████╗  ██║   ██║██║██╔██╗ ██║
    ██║  ██║██╔══╝  ╚██╗ ██╔╝██║██║╚██╗██║
    ██████╔╝███████╗ ╚████╔╝ ██║██║ ╚████║
    ╚═════╝ ╚══════╝  ╚═══╝  ╚═╝╚═╝  ╚═══╝
    --- Artificial General Intelligence v1.0.0 ---
    """)
    
    agi = None
    try:
        agi = DevinAGI()
        agi.run()
    except KeyboardInterrupt:
        logger.info("User initiated shutdown (Ctrl+C).")
    except Exception as e:
        logger.critical(f"A critical error occurred in the main loop: {e}", exc_info=True)
    finally:
        if agi:
            agi.shutdown()
