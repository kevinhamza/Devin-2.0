# Devin/modules/user_interaction_module.py
# Purpose: The central module for handling all text-based interaction
#          between the AGI and the human operator.

import logging
import sys

# Configure basic logging
logger = logging.getLogger("UserInteractionManager")
if not logger.handlers:
    h = logging.StreamHandler()
    h.setFormatter(logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s'))
    logger.addHandler(h)
    logger.setLevel(logging.INFO)

# --- ANSI Color Codes for Formatted Output ---
class Colors:
    """Container for ANSI color codes."""
    HEADER = '\033[95m'
    OKBLUE = '\033[94m'
    OKCYAN = '\033[96m'
    OKGREEN = '\033[92m'
    WARNING = '\033[93m'
    FAIL = '\033[91m'
    ENDC = '\033[0m'
    BOLD = '\033[1m'
    UNDERLINE = '\033[4m'

class UserInteractionManager:
    """
    Manages all console I/O, providing a clear and secure interface
    for the user to interact with the AGI.
    """

    def get_user_input(self, prompt: str) -> str:
        """
        Prompts the user for general text input.

        Args:
            prompt (str): The message to display to the user.

        Returns:
            str: The user's entered text.
        """
        try:
            # Use a distinct color for user prompts
            full_prompt = f"{Colors.OKCYAN}{Colors.BOLD}[PROMPT] {prompt}{Colors.ENDC} "
            response = input(full_prompt)
            return response.strip()
        except KeyboardInterrupt:
            logger.warning("\nUser interrupted input. Returning empty string.")
            return ""

    def ask_for_confirmation(self, prompt: str, is_dangerous: bool = False) -> bool:
        """
        Asks the user a yes/no question and returns a boolean.
        Defaults to 'no' for any ambiguous input.

        Args:
            prompt (str): The confirmation question to ask.
            is_dangerous (bool): If True, displays a stark warning.

        Returns:
            bool: True if the user confirms, False otherwise.
        """
        if is_dangerous:
            # Display a multi-line, high-visibility warning for dangerous actions
            warning_header = f"{Colors.FAIL}{Colors.BOLD}{'='*60}\n"
            warning_header += "!!! DANGEROUS ACTION REQUIRES CONFIRMATION !!!\n"
            warning_header += f"{'='*60}{Colors.ENDC}"
            
            full_prompt = f"\n{warning_header}\n{Colors.WARNING}{prompt}{Colors.ENDC}\n"
            prompt_suffix = f"{Colors.BOLD}Are you absolutely sure you want to proceed? (yes/no):{Colors.ENDC} "
        else:
            full_prompt = f"{Colors.WARNING}[CONFIRM] {prompt}{Colors.ENDC}"
            prompt_suffix = " (y/n): "

        try:
            response = input(full_prompt + prompt_suffix).lower().strip()
            # Fail-safe design: only explicit 'yes' or 'y' are accepted.
            if response in ['y', 'yes']:
                return True
            else:
                return False
        except KeyboardInterrupt:
            logger.warning("\nUser interrupted confirmation. Defaulting to NO.")
            return False

    def display_message(self, message: str, level: str = 'info'):
        """
        Displays a formatted message to the user.

        Args:
            message (str): The message content.
            level (str): The message level ('info', 'success', 'warning', 'error').
        """
        if level == 'info':
            print(f"{Colors.OKBLUE}[INFO] {message}{Colors.ENDC}")
        elif level == 'success':
            print(f"{Colors.OKGREEN}[SUCCESS] {message}{Colors.ENDC}")
        elif level == 'warning':
            print(f"{Colors.WARNING}[WARNING] {message}{Colors.ENDC}")
        elif level == 'error':
            print(f"{Colors.FAIL}[ERROR] {message}{Colors.ENDC}")
        else:
            print(message)

# --- Example Usage ---
if __name__ == "__main__":
    print("=========================================================")
    print("=== User Interaction Manager Demo 🗣️ ===")
    print("=========================================================")
    
    uim = UserInteractionManager()
    
    # --- 1. Demonstrate Displaying Messages ---
    print("\n--- 1. Demonstrating Message Display ---")
    uim.display_message("This is a standard informational message.")
    uim.display_message("The operation completed successfully.", level='success')
    uim.display_message("The API key is about to expire.", level='warning')
    uim.display_message("Connection to the server failed.", level='error')
    
    # --- 2. Demonstrate Getting User Input ---
    print("\n--- 2. Demonstrating General User Input ---")
    user_name = uim.get_user_input("Please enter your name:")
    uim.display_message(f"Hello, {user_name}! It's nice to meet you.", level='info')
    
    # --- 3. Demonstrate Safe Confirmation ---
    print("\n--- 3. Demonstrating a Safe Confirmation Prompt ---")
    if uim.ask_for_confirmation("Do you want to proceed with the standard operation?"):
        uim.display_message("User confirmed the safe operation.", level='success')
    else:
        uim.display_message("User denied the safe operation.", level='warning')

    # --- 4. Demonstrate Dangerous Confirmation ---
    print("\n--- 4. Demonstrating a DANGEROUS Confirmation Prompt ---")
    if uim.ask_for_confirmation("This action will format the primary hard drive.", is_dangerous=True):
        uim.display_message("User CONFIRMED the dangerous operation. (This is a demo, no action taken)", level='error')
    else:
        uim.display_message("User correctly DENIED the dangerous operation.", level='success')

    print("\n=========================================================")
    print("=== User Interaction Demo Complete ===")
    print("=========================================================")
