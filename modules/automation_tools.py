# # Devin/modules/automation_tools.py
# # Purpose: Provides a "toolbox" of low-level functions for automating
# #          desktop, browser, file system, and other computer interactions.
# # Automates tasks like scheduling and routines and all the automation of computer things that a user of computer can do 🤖🖱️

# import logging
# import time
# import os
# import random
# from pathlib import Path
# from typing import List, Dict, Any, Optional, Tuple, Union

# # Configure basic logging
# logger = logging.getLogger("AutomationTools")
# if not logger.handlers:
#     _console_handler = logging.StreamHandler()
#     _console_handler.setFormatter(logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s'))
#     logger.addHandler(_console_handler)
#     logger.setLevel(logging.INFO)


# class DesktopAutomator:
#     """
#     Conceptually automates desktop GUI interactions (mouse, keyboard, screen).
#     In a real system, this would be a wrapper around libraries like 'pyautogui' and 'pynput'.
#     """
#     def __init__(self, output_dir: str = "devin_desktop_captures"):
#         self.output_dir = Path(output_dir)
#         self.output_dir.mkdir(parents=True, exist_ok=True)
#         self.screen_width, self.screen_height = self.get_screen_size_conceptual()
#         logger.info(f"DesktopAutomator initialized. Screen size: {self.screen_width}x{self.screen_height}. Output dir: '{self.output_dir.resolve()}'")
#         logger.warning("All desktop automation actions are conceptual and require the screen to be active and unobstructed.")

#     def get_screen_size_conceptual(self) -> Tuple[int, int]:
#         """Gets the main screen resolution."""
#         logger.info("CONCEPTUAL: Using 'pyautogui.size()' to get screen dimensions.")
#         # Simulate common screen resolutions
#         return random.choice([(1920, 1080), (2560, 1440), (1366, 768)])

#     def move_mouse_to_conceptual(self, x: int, y: int, duration_sec: float = 0.5, tween: str = "easeInOutQuad") -> None:
#         """Conceptually moves the mouse to an absolute (x, y) screen coordinate."""
#         x = min(max(0, x), self.screen_width)
#         y = min(max(0, y), self.screen_height)
#         logger.info(f"CONCEPTUAL: Using 'pyautogui.moveTo({x}, {y}, duration={duration_sec}, tween=pyautogui.{tween})'")
#         time.sleep(duration_sec * 0.1) # Simulate movement time

#     def mouse_click_conceptual(self, x: Optional[int] = None, y: Optional[int] = None, button: str = 'left', clicks: int = 1, interval_sec: float = 0.1) -> None:
#         """
#         Conceptually performs a mouse click. Clicks at (x,y) if provided, otherwise at the current position.
#         """
#         if x is not None and y is not None:
#             self.move_mouse_to_conceptual(x, y, duration_sec=0.1)
#         logger.info(f"CONCEPTUAL: Using 'pyautogui.click(button='{button}', clicks={clicks}, interval={interval_sec})'")

#     def mouse_drag_conceptual(self, start_x: int, start_y: int, end_x: int, end_y: int, duration_sec: float = 1.0, button: str = 'left') -> None:
#         """Conceptually performs a mouse drag operation."""
#         self.move_mouse_to_conceptual(start_x, start_y, duration_sec=0.1)
#         logger.info(f"CONCEPTUAL: Using 'pyautogui.dragTo({end_x}, {end_y}, duration={duration_sec}, button='{button}')'")
#         time.sleep(duration_sec * 0.1)

#     def type_text_conceptual(self, text: str, interval_chars_sec: float = 0.05) -> None:
#         """Conceptually types a string of text from the keyboard."""
#         logger.info(f"CONCEPTUAL: Using 'pyautogui.write(\"{text[:30]}...\", interval={interval_chars_sec})'")

#     def press_hotkey_conceptual(self, *keys: str) -> None:
#         """Conceptually presses a combination of keys (e.g., 'ctrl', 'c')."""
#         key_str = "', '".join(keys)
#         logger.info(f"CONCEPTUAL: Using 'pyautogui.hotkey('{key_str}')'")

#     def take_screenshot_conceptual(self, region: Optional[Tuple[int, int, int, int]] = None) -> Optional[str]:
#         """
#         Conceptually takes a screenshot of the whole screen or a specific region.
#         Region is a tuple of (left, top, width, height).
#         """
#         output_filename = f"screenshot_{time.strftime('%Y%m%d_%H%M%S')}.png"
#         output_path = self.output_dir / output_filename
        
#         if region:
#             logger.info(f"CONCEPTUAL: Using 'pyautogui.screenshot({output_path}, region={region})'")
#         else:
#             logger.info(f"CONCEPTUAL: Using 'pyautogui.screenshot({output_path})'")
        
#         output_path.touch() # Create dummy file
#         logger.info(f"  Screenshot conceptually saved to '{output_path}'")
#         return str(output_path)


# class BrowserAutomator:
#     """
#     Conceptually automates web browser interactions.
#     In a real system, this would be a wrapper around a library like 'Selenium' or 'Playwright'.
#     """
#     def __init__(self, browser_type: str = "chrome"):
#         self.browser_type = browser_type
#         self.driver_placeholder: Optional[Dict[str, Any]] = None
#         logger.info(f"BrowserAutomator initialized for conceptual '{browser_type}' control.")

#     def open_browser_conceptual(self, headless: bool = False) -> bool:
#         """Conceptually launches and initializes a web browser driver."""
#         if self.driver_placeholder:
#             logger.warning(f"Browser driver already seems to be active.")
#             return True
#         logger.info(f"CONCEPTUAL: Initializing '{self.browser_type}' driver (headless={headless}) using 'selenium.webdriver.{self.browser_type.capitalize()}()'")
#         self.driver_placeholder = {
#             "id": f"driver_{uuid.uuid4().hex[:8]}",
#             "headless": headless,
#             "current_url": None,
#             "current_title": None,
#             "window_handle": "handle_main"
#         }
#         return True

#     def close_browser_conceptual(self) -> None:
#         """Conceptually closes the browser and quits the driver."""
#         if not self.driver_placeholder:
#             logger.warning("No active browser driver to close.")
#             return
#         logger.info(f"CONCEPTUAL: Using 'driver.quit()' to close browser and session '{self.driver_placeholder['id']}'.")
#         self.driver_placeholder = None

#     def navigate_to_url_conceptual(self, url: str) -> None:
#         """Conceptually navigates the browser to a specific URL."""
#         if not self.driver_placeholder:
#             logger.error("Cannot navigate: Browser not open.")
#             return
#         logger.info(f"CONCEPTUAL: Using 'driver.get(\"{url}\")'")
#         self.driver_placeholder["current_url"] = url
#         self.driver_placeholder["current_title"] = f"Simulated Page Title for {url}"
#         time.sleep(0.2) # Simulate page load

#     def find_element_conceptual(self, by: str, value: str) -> Dict[str, Any]:
#         """
#         Conceptually finds an element on the page.
#         'by' can be 'id', 'name', 'xpath', 'css_selector', 'link_text'.
#         """
#         if not self.driver_placeholder:
#             logger.error("Cannot find element: Browser not open.")
#             return {"status": "error", "message": "Browser not open."}
        
#         logger.info(f"CONCEPTUAL: Using 'driver.find_element(By.{by.upper()}, \"{value}\")'")
#         # Simulate finding an element
#         return {
#             "status": "success",
#             "element_id": f"elem_{uuid.uuid4().hex[:6]}",
#             "tag_name": random.choice(["input", "button", "a", "div"]),
#             "text": "Simulated Element Text"
#         }

#     def click_element_conceptual(self, by: str, value: str) -> None:
#         """Conceptually finds an element and clicks it."""
#         element = self.find_element_conceptual(by, value)
#         if element["status"] == "success":
#             logger.info(f"CONCEPTUAL: Using 'element.click()' on element found by {by}='{value}' (ID: {element['element_id']})")
#         else:
#             logger.error(f"Could not click element; find operation failed.")

#     def type_into_element_conceptual(self, by: str, value: str, text_to_type: str) -> None:
#         """Conceptually finds an input element and types text into it."""
#         element = self.find_element_conceptual(by, value)
#         if element["status"] == "success" and element["tag_name"] in ["input", "textarea"]:
#             logger.info(f"CONCEPTUAL: Using 'element.send_keys(\"{text_to_type[:30]}...\")' on element found by {by}='{value}'")
#         else:
#             logger.error(f"Could not type into element; find operation failed or element is not an input field.")

#     def get_page_source_conceptual(self) -> Optional[str]:
#         """Conceptually gets the full HTML source of the current page."""
#         if not self.driver_placeholder:
#             logger.error("Cannot get page source: Browser not open.")
#             return None
#         logger.info("CONCEPTUAL: Using 'driver.page_source' property.")
#         return f"<html><head><title>{self.driver_placeholder['current_title']}</title></head><body><h1>Simulated Page for {self.driver_placeholder['current_url']}</h1></body></html>"

#     def execute_script_conceptual(self, script: str) -> Any:
#         """Conceptually executes JavaScript in the context of the current page."""
#         if not self.driver_placeholder:
#             logger.error("Cannot execute script: Browser not open.")
#             return None
#         logger.info(f"CONCEPTUAL: Using 'driver.execute_script(\"{script[:50]}...\")'")
#         # Simulate a return value from the script
#         if "return" in script:
#             return "Simulated JavaScript execution result"
#         return None

# import logging # Already imported in Part 1
# import time # Already imported in Part 1
# import os # Already imported in Part 1
# import random # Already imported in Part 1
# import uuid # Imported for this part
# from pathlib import Path # Already imported in Part 1
# from typing import List, Dict, Any, Optional, Tuple, Union, Callable # Updated for this part

# class FileSystemAutomator:
#     """
#     Conceptually automates advanced file system operations.
#     In a real system, this would wrap libraries like 'os', 'shutil', 'glob', and 'watchdog'.
#     """
#     def __init__(self):
#         logger.info("FileSystemAutomator initialized.")

#     def find_files_conceptual(self, start_path: str, name_pattern: str) -> List[str]:
#         """
#         Conceptually finds files matching a pattern in a directory tree.
#         Real-world equivalent: `glob.glob(f'{start_path}/**/{name_pattern}', recursive=True)`
#         """
#         logger.info(f"CONCEPTUAL: Using 'glob' to search for '{name_pattern}' in '{start_path}'.")
#         # Simulate finding some files
#         simulated_parent = Path(start_path)
#         return [
#             str(simulated_parent / "subdir1" / f"log_{random.randint(100,200)}.txt"),
#             str(simulated_parent / "subdir2" / "logs" / f"log_{random.randint(201,300)}.txt"),
#         ]

#     def move_file_conceptual(self, source_path: str, destination_path: str) -> None:
#         """
#         Conceptually moves or renames a file.
#         Real-world equivalent: `shutil.move(source, destination)`
#         """
#         logger.info(f"CONCEPTUAL: Using 'shutil.move()' from '{source_path}' to '{destination_path}'.")

#     def copy_directory_conceptual(self, source_dir: str, destination_dir: str) -> None:
#         """
#         Conceptually copies an entire directory tree.
#         Real-world equivalent: `shutil.copytree(source, destination)`
#         """
#         logger.info(f"CONCEPTUAL: Using 'shutil.copytree()' from '{source_dir}' to '{destination_dir}'.")

#     def watch_directory_conceptual(self, path_to_watch: str, event_handler: Callable, duration_sec: int) -> None:
#         """
#         Conceptually watches a directory for changes for a set duration.
#         Real-world equivalent: Using the 'watchdog' library.
#         """
#         logger.info(f"CONCEPTUAL: Using 'watchdog' to monitor '{path_to_watch}' for {duration_sec} seconds.")
#         logger.info("  Starting conceptual watch...")
#         # Simulate a file creation event during the watch period
#         time.sleep(duration_sec * 0.1)
#         conceptual_event = {"event_type": "created", "src_path": str(Path(path_to_watch) / "new_report.pdf")}
#         logger.info(f"  ...Conceptual event detected: {conceptual_event}")
#         event_handler(conceptual_event)
#         logger.info("  ...Conceptual watch finished.")


# class ProcessManager:
#     """
#     Conceptually manages system processes (applications).
#     In a real system, this would wrap 'subprocess' and 'psutil'.
#     """
#     def __init__(self):
#         self.running_processes_conceptual: Dict[int, Dict] = {}
#         logger.info("ProcessManager initialized.")

#     def start_process_conceptual(self, command: Union[str, List[str]]) -> int:
#         """
#         Conceptually starts a new process.
#         Real-world equivalent: `subprocess.Popen(command)`
#         """
#         command_str = command if isinstance(command, str) else ' '.join(command)
#         logger.info(f"CONCEPTUAL: Using 'subprocess.Popen' to start process: '{command_str}'")
#         pid = random.randint(10000, 40000)
#         self.running_processes_conceptual[pid] = {"command": command_str, "status": "running"}
#         logger.info(f"  Process started with conceptual PID: {pid}")
#         return pid

#     def terminate_process_conceptual(self, pid: int) -> bool:
#         """
#         Conceptually terminates a process by its PID.
#         Real-world equivalent: `psutil.Process(pid).terminate()`
#         """
#         if pid not in self.running_processes_conceptual:
#             logger.error(f"Cannot terminate conceptual process: PID {pid} not found.")
#             return False
#         logger.info(f"CONCEPTUAL: Using 'psutil.Process({pid}).terminate()' to stop process.")
#         self.running_processes_conceptual[pid]["status"] = "terminated"
#         return True

#     def is_process_running_conceptual(self, pid: int) -> bool:
#         """
#         Conceptually checks if a process is still running.
#         Real-world equivalent: `psutil.pid_exists(pid)`
#         """
#         logger.info(f"CONCEPTUAL: Using 'psutil.pid_exists({pid})' to check process status.")
#         return self.running_processes_conceptual.get(pid, {}).get("status") == "running"


# class TaskScheduler:
#     """
#     Conceptually schedules tasks to be run at specific times or intervals.
#     In a real system, this could wrap 'schedule', 'apscheduler', or interact with OS schedulers.
#     """
#     def __init__(self):
#         self._jobs = []
#         logger.info("TaskScheduler initialized.")

#     def schedule_once_at_conceptual(self, task_function: Callable, run_time: datetime, *args, **kwargs) -> str:
#         """Conceptually schedules a function to run once at a specific time."""
#         job_id = f"job_{uuid.uuid4().hex[:6]}"
#         logger.info(f"CONCEPTUAL SCHEDULER: Using 'schedule' to run task '{task_function.__name__}' once at {run_time.strftime('%Y-%m-%d %H:%M:%S')}. Job ID: {job_id}")
#         self._jobs.append({"id": job_id, "func": task_function, "time": run_time})
#         return job_id

#     def schedule_every_interval_conceptual(self, task_function: Callable, interval_minutes: int, *args, **kwargs) -> str:
#         """Conceptually schedules a function to run repeatedly."""
#         job_id = f"job_{uuid.uuid4().hex[:6]}"
#         logger.info(f"CONCEPTUAL SCHEDULER: Using 'schedule' to run task '{task_function.__name__}' every {interval_minutes} minutes. Job ID: {job_id}")
#         self._jobs.append({"id": job_id, "func": task_function, "interval": interval_minutes})
#         return job_id
        
#     def run_pending_conceptual(self) -> None:
#         """
#         Simulates checking and running scheduled jobs. In a real app, this would be in a loop.
#         """
#         logger.info("CONCEPTUAL SCHEDULER: Checking for pending jobs to run...")
#         # This is a very simple simulation. A real library manages the event loop.
#         if self._jobs:
#             job_to_run = self._jobs.pop(0) # Just run the first one for demo
#             logger.info(f"  -> Running job '{job_to_run['id']}': executing '{job_to_run['func'].__name__}'")
#             job_to_run['func']()
#         else:
#             logger.info("  -> No pending jobs.")

# # --- Example Usage Tying All Tools Together ---
# if __name__ == "__main__":
#     print("================================================================")
#     print("=== Automation Tools Module Prototype (Full Toolbox Demo) 🤖 ===")
#     print("================================================================")

#     # Initialize all our conceptual automation tools
#     desktop = DesktopAutomator()
#     browser = BrowserAutomator(browser_type="firefox")
#     file_system = FileSystemAutomator()
#     process_manager = ProcessManager()
#     scheduler = TaskScheduler()

#     # --- Define a complex, multi-faceted automation workflow ---
#     def automated_report_workflow():
#         print("\n--- STARTING AUTOMATED WORKFLOW ---")
        
#         # 1. Start a local server process (conceptual)
#         print("\n[Step 1: Process Management]")
#         server_pid = process_manager.start_process_conceptual(["python", "-m", "http.server", "8080"])
#         time.sleep(0.2)

#         # 2. Use the browser to navigate to the local server and get data
#         print("\n[Step 2: Browser Automation]")
#         browser.open_browser_conceptual(headless=True)
#         browser.navigate_to_url_conceptual("http://localhost:8080")
#         time.sleep(0.2)
#         page_source = browser.get_page_source_conceptual()
#         print(f"  Got page source from local server (first 50 chars): '{page_source[:50]}...'")
        
#         # 3. Use desktop automation to take a screenshot of a specific area (e.g., a "report" area)
#         print("\n[Step 3: Desktop Automation]")
#         # Coordinates are for a 1920x1080 screen conceptually
#         report_area = (100, 150, 800, 600)
#         screenshot_path = desktop.take_screenshot_conceptual(region=report_area)
#         print(f"  Took screenshot of 'report area', saved to '{screenshot_path}'")
        
#         # 4. Stop the local server process
#         print("\n[Step 4: Process Management]")
#         process_manager.terminate_process_conceptual(server_pid)
#         print(f"  Process with PID {server_pid} terminated: {not process_manager.is_process_running_conceptual(server_pid)}")
        
#         # 5. Clean up by moving the screenshot to an 'archive' folder
#         print("\n[Step 5: File System Automation]")
#         archive_path = Path("./archive")
#         archive_path.mkdir(exist_ok=True)
#         if screenshot_path:
#             file_system.move_file_conceptual(screenshot_path, str(archive_path / Path(screenshot_path).name))
        
#         # Close the browser
#         browser.close_browser_conceptual()
        
#         print("\n--- AUTOMATED WORKFLOW COMPLETE ---")

#     # --- Define another simple task for the scheduler ---
#     def scheduled_cleanup_task():
#         print("\n*** EXECUTING SCHEDULED CLEANUP TASK ***")
#         log_files = file_system.find_files_conceptual("./logs", "*.log")
#         print(f"  Conceptual cleanup task found {len(log_files)} log files to archive.")
#         print("*** CLEANUP TASK COMPLETE ***")


#     # --- Schedule the main workflow and a recurring task ---
#     print("\n[Main Section: Scheduling Tasks]")
#     # Schedule the main workflow to run "now" (it will be picked up by the first run_pending)
#     workflow_run_time = datetime.now() + timedelta(seconds=1)
#     scheduler.schedule_once_at_conceptual(automated_report_workflow, workflow_run_time)
    
#     # Schedule the cleanup task to run every 10 minutes (conceptually)
#     scheduler.schedule_every_interval_conceptual(scheduled_cleanup_task, interval_minutes=10)

#     # --- Simulate the scheduler's main loop checking for jobs to run ---
#     # In a real app, this would be `while True: scheduler.run_pending(); time.sleep(1)`
#     print("\n[Main Section: Simulating Scheduler Running Pending Jobs]")
#     scheduler.run_pending_conceptual() # This will run the `automated_report_workflow`
    
#     print("\n[Main Section: Simulating another check later...]")
#     scheduler.run_pending_conceptual() # This will run the `scheduled_cleanup_task`

#     print("\n================================================================")
#     print("=== Automation Tools Prototype Complete ===")
#     print("================================================================")


# Devin/modules/automation_tools.py
# Purpose: Provides a "toolbox" of live-fire functions for automating
#          desktop, browser, file system, and other computer interactions.

import logging
import time
import os
import shutil
import glob
import subprocess
from pathlib import Path
from typing import List, Optional, Tuple

try:
    import pyautogui
    import psutil
    from selenium import webdriver
    from selenium.webdriver.common.by import By
    from selenium.webdriver.chrome.service import Service as ChromeService
    from webdriver_manager.chrome import ChromeDriverManager
    DEPS_AVAILABLE = True
except ImportError as e:
    DEPS_AVAILABLE = False
    _import_error = e

# Configure basic logging
logger = logging.getLogger("AutomationTools")
if not logger.handlers:
    _console_handler = logging.StreamHandler()
    _console_handler.setFormatter(logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s'))
    logger.addHandler(_console_handler)
    logger.setLevel(logging.INFO)


class DesktopAutomator:
    """Automates desktop GUI interactions using 'pyautogui'."""
    def __init__(self, output_dir: str = "devin_desktop_captures"):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.screen_width, self.screen_height = pyautogui.size()
        logger.info(f"DesktopAutomator initialized. Screen size: {self.screen_width}x{self.screen_height}.")

    def move_mouse_to(self, x: int, y: int, duration_sec: float = 0.5):
        pyautogui.moveTo(x, y, duration=duration_sec, tween=pyautogui.easeInOutQuad)

    def mouse_click(self, button: str = 'left', clicks: int = 1, interval_sec: float = 0.1):
        pyautogui.click(button=button, clicks=clicks, interval=interval_sec)

    def type_text(self, text: str, interval_chars_sec: float = 0.05):
        pyautogui.write(text, interval=interval_chars_sec)

    def take_screenshot(self) -> str:
        output_path = self.output_dir / f"screenshot_{time.strftime('%Y%m%d_%H%M%S')}.png"
        pyautogui.screenshot(str(output_path))
        logger.info(f"Screenshot saved to '{output_path}'")
        return str(output_path)


class WebAutomator:
    """Automates web browser interactions using 'Selenium'."""
    def __init__(self, browser_type: str = "chrome"):
        self.browser_type = browser_type
        self.driver: Optional[webdriver.Chrome] = None

    def open_browser(self, headless: bool = True):
        logger.info(f"Initializing '{self.browser_type}' driver (headless={headless})...")
        options = webdriver.ChromeOptions()
        if headless:
            options.add_argument("--headless")
        self.driver = webdriver.Chrome(service=ChromeService(ChromeDriverManager().install()), options=options)
        logger.info("Browser opened successfully.")

    def close_browser(self):
        if self.driver:
            self.driver.quit()
            self.driver = None
            logger.info("Browser closed.")

    def navigate_to_url(self, url: str):
        if not self.driver: raise ConnectionError("Browser is not open.")
        self.driver.get(url)
        logger.info(f"Navigated to URL: {url}")

    def find_element_and_click(self, by: By, value: str):
        if not self.driver: raise ConnectionError("Browser is not open.")
        element = self.driver.find_element(by, value)
        element.click()
        logger.info(f"Clicked element found by {by}='{value}'")


class FileSystemAutomator:
    """Automates file system operations."""
    def find_files(self, start_path: str, name_pattern: str) -> List[str]:
        return glob.glob(f'{start_path}/**/{name_pattern}', recursive=True)
    
    def move_file(self, source_path: str, destination_path: str):
        shutil.move(source_path, destination_path)
        logger.info(f"Moved file from '{source_path}' to '{destination_path}'.")


class ProcessManager:
    """Manages system processes using 'subprocess' and 'psutil'."""
    def start_process(self, command: List[str]) -> psutil.Process:
        proc = subprocess.Popen(command, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        p = psutil.Process(proc.pid)
        logger.info(f"Started process '{' '.join(command)}' with PID: {p.pid}")
        return p

    def terminate_process(self, pid: int):
        try:
            p = psutil.Process(pid)
            p.terminate()
            logger.info(f"Terminated process with PID: {pid}")
        except psutil.NoSuchProcess:
            logger.warning(f"Process with PID {pid} not found. May have already terminated.")

# --- Example Usage Tying All Tools Together ---
if __name__ == "__main__":
    print("=========================================================")
    print("=== Integrated Automation Tools Prototype 🤖🖱️ ===")
    print("=========================================================")

    if not DEPS_AVAILABLE:
        print(f"\nERROR: A core dependency is missing. Please run 'pip install pyautogui selenium psutil webdriver-manager'.")
        print(f"Import Error: {_import_error}")
    else:
        # 1. Initialize all automation tools
        desktop = DesktopAutomator()
        browser = BrowserAutomator()
        file_system = FileSystemAutomator()
        process_manager = ProcessManager()
        server_process = None

        print("--- Starting a complex automation workflow ---")
        try:
            # 2. Start a local web server as a background process
            print("\n[Step 1] Starting a local Python web server...")
            server_process = process_manager.start_process(["python", "-m", "http.server", "8000"])
            time.sleep(2) # Give server time to start up

            # 3. Open a browser and navigate to the local server
            print("\n[Step 2] Opening a headless browser to interact with the server...")
            browser.open_browser(headless=True)
            browser.navigate_to_url("http://localhost:8000")
            
            # 4. Use desktop automation to take a screenshot of the whole screen
            print("\n[Step 3] Taking a screenshot of the desktop...")
            screenshot_path_str = desktop.take_screenshot()
            
            # 5. Use file system automation to move the screenshot
            print("\n[Step 4] Moving the screenshot to a new directory...")
            archive_dir = Path("./automation_archive")
            archive_dir.mkdir(exist_ok=True)
            screenshot_path = Path(screenshot_path_str)
            file_system.move_file(str(screenshot_path), str(archive_dir / screenshot_path.name))

        except Exception as e:
            logger.error(f"An error occurred during the automation workflow: {e}", exc_info=True)
        finally:
            # 6. Clean up: close the browser and terminate the server process
            print("\n[Step 5] Cleaning up resources...")
            browser.close_browser()
            if server_process:
                process_manager.terminate_process(server_process.pid)
            
            print("\n--- Automation workflow complete! ---")
            
    print("\n=========================================================")
    print("=== Automation Tools Prototype Complete ===")
    print("=========================================================")
