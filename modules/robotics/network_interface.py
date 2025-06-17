# Devin/modules/robotics/network_interface.py
# Purpose: Provides a decoupled, intra-robot communication system using a
#          Publish/Subscribe (Pub/Sub) message bus pattern.

import logging
import threading
import queue
import time
from collections import defaultdict
from typing import Any, Callable, Dict, List

# Configure basic logging
logger = logging.getLogger("RoboticsNetwork")
if not logger.handlers:
    _console_handler = logging.StreamHandler()
    _console_handler.setFormatter(logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s'))
    logger.addHandler(_console_handler)
    logger.setLevel(logging.INFO)

class NetworkInterface:
    """
    A thread-safe, singleton Publish/Subscribe message bus for the robotics system.
    """
    _instance = None
    _lock = threading.Lock()

    def __new__(cls, *args, **kwargs):
        if not cls._instance:
            with cls._lock:
                if not cls._instance:
                    cls._instance = super(NetworkInterface, cls).__new__(cls)
        return cls._instance

    def __init__(self):
        # This check ensures __init__ is only run once for the singleton instance
        if not hasattr(self, '_initialized'):
            with self._lock:
                if not hasattr(self, '_initialized'):
                    self.subscribers: Dict[str, List[Callable[[Any], None]]] = defaultdict(list)
                    self.message_queue = queue.Queue()
                    self.processing_thread: threading.Thread = None
                    self._stop_event = threading.Event()
                    self._initialized = True
                    logger.info("RoboticsNetwork Interface (Singleton) initialized.")

    def start(self):
        """Starts the background message processing thread."""
        if self.processing_thread and self.processing_thread.is_alive():
            logger.warning("Network interface is already running.")
            return

        self._stop_event.clear()
        self.processing_thread = threading.Thread(target=self._message_processor_loop, daemon=True)
        self.processing_thread.start()
        logger.info("Pub/Sub message processor thread started.")

    def stop(self):
        """Stops the message processing thread gracefully."""
        if not self.processing_thread or not self.processing_thread.is_alive():
            logger.info("Network interface is not running.")
            return
            
        logger.info("Stopping network interface...")
        self.message_queue.put(None) # Sentinel value to stop the loop
        self.processing_thread.join(timeout=2.0)
        self._stop_event.set()
        logger.info("Network interface stopped.")

    def subscribe(self, topic: str, callback: Callable[[Any], None]):
        """
        Subscribes a callback function to a specific topic.

        Args:
            topic (str): The topic to subscribe to (e.g., '/sensors/imu').
            callback (Callable[[Any], None]): The function to call with the message
                                              when data is published on the topic.
        """
        logger.info(f"New subscription for topic '{topic}' by callback '{callback.__name__}'.")
        self.subscribers[topic].append(callback)

    def publish(self, topic: str, message: Any):
        """
        Publishes a message to a specific topic. This method is non-blocking.

        Args:
            topic (str): The topic to publish the message on.
            message (Any): The data/message payload.
        """
        if self._stop_event.is_set():
            logger.warning("Interface is stopped. Cannot publish.")
            return
            
        logger.debug(f"Publishing message to topic '{topic}'.")
        self.message_queue.put((topic, message))

    def _message_processor_loop(self):
        """The background worker that distributes messages to subscribers."""
        while not self._stop_event.is_set():
            try:
                message_item = self.message_queue.get()
                
                # Check for the stop signal
                if message_item is None:
                    break

                topic, message = message_item
                
                # Find all callbacks subscribed to this topic
                if topic in self.subscribers:
                    logger.debug(f"Distributing message on '{topic}' to {len(self.subscribers[topic])} subscribers.")
                    for callback in self.subscribers[topic]:
                        try:
                            # Execute the callback with the message
                            callback(message)
                        except Exception as e:
                            logger.error(f"Error in subscriber callback '{callback.__name__}' for topic '{topic}': {e}")
            except Exception as e:
                 logger.error(f"Error in message processing loop: {e}")


# --- Example Usage ---
if __name__ == "__main__":
    print("=========================================================")
    print("=== Robotics Network Interface Prototype 🌐 === ")
    print("=========================================================")

    # 1. Get the singleton instance of the network interface and start it
    bus = NetworkInterface()
    bus.start()

    # 2. Define some subscriber callbacks (representing different modules)
    def vision_processor_subscriber(image_data):
        """This function would run in a computer vision module."""
        print(f"  [Vision Processor] Received image frame with timestamp {image_data['timestamp']}. Processing for objects...")

    def logger_subscriber(data):
        """This function would run in the data logger module."""
        print(f"  [Data Logger]      Received image frame with timestamp {data['timestamp']}. Writing to log file...")

    # 3. Subscribe the callbacks to a topic
    print("\n--- Subscribing modules to the '/sensors/camera' topic ---")
    bus.subscribe("/sensors/camera", vision_processor_subscriber)
    bus.subscribe("/sensors/camera", logger_subscriber)

    # 4. Define and start a publisher (representing a hardware driver)
    def camera_publisher(stop_event):
        """This function simulates a camera driver publishing images."""
        while not stop_event.is_set():
            # Simulate capturing an image
            image_message = {
                "timestamp": time.time(),
                "format": "bgr8",
                "resolution": (1920, 1080),
                "data": "<conceptual_numpy_array_bytes>"
            }
            print(f"\n[Camera Driver] Publishing new image frame...")
            bus.publish("/sensors/camera", image_message)
            time.sleep(2) # Publish at 0.5 Hz

    print("\n--- Starting a simulated Camera Publisher in a background thread ---")
    publisher_stop_event = threading.Event()
    publisher_thread = threading.Thread(target=camera_publisher, args=(publisher_stop_event,), daemon=True)
    publisher_thread.start()
    
    # 5. Let the simulation run for a few seconds
    try:
        time.sleep(5)
    except KeyboardInterrupt:
        pass
    finally:
        # 6. Cleanly shut down the system
        print("\n\n--- Shutting down the system ---")
        publisher_stop_event.set()
        publisher_thread.join(timeout=1.0)
        bus.stop()

    print("\n=========================================================")
    print("=== Network Interface Prototype Complete ===")
    print("=========================================================")
