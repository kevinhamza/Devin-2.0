# Devin/modules/robotics/remote_control.py
# Purpose: Provides a real-time, non-blocking remote control system for a robot's
#          base movement using the keyboard of the host computer.

import logging
import threading
import time
from typing import Optional, Any, Dict

# --- Dependency Installation Notes ---
# This module requires the 'pynput' library for cross-platform keyboard monitoring.
# pip install pynput
try:
    from pynput import keyboard
    PYNPUT_AVAILABLE = True
except ImportError:
    PYNPUT_AVAILABLE = False
    keyboard = None

# --- Conceptual Placeholder for the Motor Controller ---
# In a real system, you would import the actual MotorController class
# from .motor_control import MotorController
class MotorController:
    """A conceptual placeholder for the actual MotorController."""
    def set_base_velocity(self, linear_vel: float, angular_vel: float):
        # This is where the command to the robot's hardware would go.
        # We simulate it by logging the intended action.
        logger.debug(f"[MotorController] Setting base velocity: linear={linear_vel:.2f} m/s, angular={angular_vel:.2f} rad/s")
    def stop_base(self):
        logger.info("[MotorController] STOPPING ALL MOVEMENT.")
        self.set_base_velocity(0.0, 0.0)
# --- End of Conceptual Placeholder ---


# Configure basic logging
logger = logging.getLogger("RemoteControl")
if not logger.handlers:
    _console_handler = logging.StreamHandler()
    _console_handler.setFormatter(logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s'))
    logger.addHandler(_console_handler)
    logger.setLevel(logging.INFO)


class RemoteControl:
    """
    Manages a real-time, keyboard-based remote control session for a robot.
    """
    def __init__(self, motor_controller: MotorController, move_speed: float = 0.5, turn_speed: float = 1.0):
        """
        Initializes the RemoteControl system.

        Args:
            motor_controller (MotorController): An instance of the low-level motor controller.
            move_speed (float): The default linear speed in meters/second.
            turn_speed (float): The default angular speed in radians/second.
        """
        if not PYNPUT_AVAILABLE:
            raise ImportError("The 'pynput' library is required for remote control. Please run 'pip install pynput'.")

        self.motors = motor_controller
        self.move_speed = move_speed
        self.turn_speed = turn_speed

        self.linear_velocity = 0.0
        self.angular_velocity = 0.0
        self.keys_pressed = set()

        self._control_thread: Optional[threading.Thread] = None
        self._listener_thread: Optional[keyboard.Listener] = None
        self._stop_event = threading.Event()
        self._lock = threading.Lock()

        logger.info("RemoteControl initialized.")

    def _update_velocity(self):
        """Calculates desired velocity based on the set of currently pressed keys."""
        with self._lock:
            # Reset velocities
            self.linear_velocity = 0.0
            self.angular_velocity = 0.0

            # Set linear velocity based on W/S keys
            if 'w' in self.keys_pressed:
                self.linear_velocity = self.move_speed
            elif 's' in self.keys_pressed:
                self.linear_velocity = -self.move_speed

            # Set angular velocity based on A/D keys
            if 'a' in self.keys_pressed:
                self.angular_velocity = self.turn_speed
            elif 'd' in self.keys_pressed:
                self.angular_velocity = -self.turn_speed

    def _on_press(self, key):
        """Callback function for pynput key press events."""
        try:
            # Get the character of the key
            key_char = key.char.lower()
            if key_char in ['w', 'a', 's', 'd']:
                with self._lock:
                    self.keys_pressed.add(key_char)
                self._update_velocity()
        except AttributeError:
            # This handles special keys like Key.shift, Key.esc
            if key == keyboard.Key.esc:
                logger.info("Escape key pressed. Shutting down remote control.")
                self.stop_control()
            # You can add more special key handling here if needed

    def _on_release(self, key):
        """Callback function for pynput key release events."""
        try:
            key_char = key.char.lower()
            if key_char in self.keys_pressed:
                with self._lock:
                    self.keys_pressed.discard(key_char)
                self._update_velocity()
        except AttributeError:
            pass # Ignore special key releases for this simple control scheme

    def _control_loop(self):
        """
        The background thread that continuously sends velocity commands to the motors.
        """
        logger.info("Control loop thread started.")
        while not self._stop_event.is_set():
            with self._lock:
                linear = self.linear_velocity
                angular = self.angular_velocity

            # Only send a command if there is movement, to avoid flooding the bus
            if linear != 0.0 or angular != 0.0:
                self.motors.set_base_velocity(linear, angular)
            else:
                # When no movement keys are pressed, ensure the robot stops.
                # This check prevents a stop command from being sent on every single loop tick.
                # We could add a simple state check to only send it once.
                self.motors.stop_base()

            time.sleep(0.1) # Send commands at 10Hz

        # Final command to ensure the robot is stopped upon exit
        self.motors.stop_base()
        logger.info("Control loop thread stopped.")

    def start_control(self):
        """Starts the remote control session (keyboard listener and control loop)."""
        if self._listener_thread and self._listener_thread.is_alive():
            logger.warning("Remote control is already active.")
            return

        logger.info("Starting remote control session.")
        self._stop_event.clear()

        # Start the control loop that sends commands to the robot
        self._control_thread = threading.Thread(target=self._control_loop, daemon=True)
        self._control_thread.start()

        # Start the non-blocking keyboard listener
        self._listener_thread = keyboard.Listener(
            on_press=self._on_press,
            on_release=self._on_release
        )
        self._listener_thread.start()
        
        print("\n--- Remote Control Active ---")
        print("Use WASD keys to move the robot.")
        print("  W: Forward")
        print("  S: Backward")
        print("  A: Turn Left")
        print("  D: Turn Right")
        print("Press 'ESC' to stop remote control.")
        print("-----------------------------")


    def stop_control(self):
        """Stops the remote control session."""
        if not self._listener_thread:
            logger.info("Remote control is not running.")
            return

        logger.info("Stopping remote control session...")
        self._stop_event.set()

        if self._listener_thread:
            self._listener_thread.stop()
            self._listener_thread.join()
            self._listener_thread = None
        
        if self._control_thread:
            self._control_thread.join(timeout=1.0)
            self._control_thread = None
            
        logger.info("Remote control session stopped.")

# --- Example Usage ---
if __name__ == "__main__":
    print("=========================================================")
    print("=== Remote Control Module Prototype ⌨️🕹️ ===")
    print("=========================================================")
    
    if not PYNPUT_AVAILABLE:
        print("\n'pynput' library not found. Please run 'pip install pynput' to run this interactive demo.")
    else:
        # 1. Initialize the motor controller (conceptual) and the remote controller
        motors = MotorController()
        remote_controller = RemoteControl(motor_controller=motors)

        # 2. Start the remote control session
        remote_controller.start_control()

        # 3. The main program can now do other things while the remote control
        #    runs in the background. We'll just wait here until ESC is pressed.
        # The listener thread will call `stop_control` when ESC is pressed.
        if remote_controller._listener_thread:
            remote_controller._listener_thread.join()
        
        print("\n--- Main Program ---")
        print("Remote control has been stopped.")
        print("The main program can now continue or exit.")

    print("\n=========================================================")
    print("=== Remote Control Prototype Complete ===")
    print("=========================================================")
