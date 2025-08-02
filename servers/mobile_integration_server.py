# Devin/servers/mobile_integration_server.py
# Purpose: A microservice that provides a REST API to interact with and
#          control connected Android devices via the Android Debug Bridge (ADB).

import logging
import threading
import time
from pathlib import Path
from io import BytesIO

try:
    from flask import Flask, request, jsonify, send_file
    from ppadb.client import Client as AdbClient
    from ppadb.device import Device
    DEVIN_CORE_AVAILABLE = True
except ImportError as e:
    DEVIN_CORE_AVAILABLE = False
    _import_error = e

# Configure basic logging
logger = logging.getLogger("MobileIntegrationServer")
if not logger.handlers:
    _console_handler = logging.StreamHandler()
    _console_handler.setFormatter(logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s'))
    logger.addHandler(_console_handler)
    logger.setLevel(logging.INFO)


class MobileIntegrationServer:
    """
    Wraps a Flask application to provide a REST API for ADB.
    """
    def __init__(self, adb_host: str = "127.0.0.1", adb_port: int = 5037):
        if not DEVIN_CORE_AVAILABLE:
            raise ImportError(f"A core Devin module is missing. Error: {_import_error}")

        # --- Initialize ADB Client ---
        try:
            self.adb_client = AdbClient(host=adb_host, port=adb_port)
            self.adb_client.version() # Check connection
            logger.info("Successfully connected to ADB server.")
        except Exception as e:
            logger.error("Could not connect to ADB server. Is it running? Is a device connected with USB debugging enabled?")
            raise ConnectionError(f"ADB connection failed: {e}")

        # --- Initialize Flask App ---
        self.app = Flask(__name__)
        self._register_routes()

    def _get_device(self, serial: str) -> Device:
        """Helper to get a device object by its serial number."""
        devices = self.adb_client.devices()
        for device in devices:
            if device.serial == serial:
                return device
        return None

    def _register_routes(self):
        """Defines the API endpoints for the server."""
        
        @self.app.route("/devices", methods=["GET"])
        def list_devices():
            devices = self.adb_client.devices()
            return jsonify([{"serial": d.serial, "status": d.get_state()} for d in devices])

        @self.app.route("/<serial>/shell", methods=["POST"])
        def shell_command(serial: str):
            device = self._get_device(serial)
            if not device: return jsonify({"error": "Device not found"}), 404
            
            data = request.get_json()
            if not data or "command" not in data:
                return jsonify({"error": "Missing 'command' in request."}), 400

            output = device.shell(data['command'])
            return jsonify({"output": output})

        @self.app.route("/<serial>/screenshot", methods=["GET"])
        def screenshot(serial: str):
            device = self._get_device(serial)
            if not device: return jsonify({"error": "Device not found"}), 404
            
            result = device.screencap()
            return send_file(BytesIO(result), mimetype='image/png')
            
        @self.app.route("/<serial>/tap", methods=["POST"])
        def tap(serial: str):
            device = self._get_device(serial)
            if not device: return jsonify({"error": "Device not found"}), 404
            
            data = request.get_json()
            if not data or "x" not in data or "y" not in data:
                return jsonify({"error": "Missing 'x' and 'y' coordinates."}), 400
            
            device.input_tap(data['x'], data['y'])
            return jsonify({"status": "success", "message": f"Tapped screen at ({data['x']}, {data['y']})"})

    def run(self, host: str = '127.0.0.1', port: int = 5006):
        """Starts the Flask web server."""
        logger.warning(f"Starting Mobile Integration Server on http://{host}:{port}")
        self.app.run(host=host, port=port)

# --- Example Usage ---
def run_client_demo():
    """A simple client to demonstrate interacting with the running server."""
    import requests
    
    SERVER_URL = "http://127.0.0.1:5006"
    
    # --- 1. List connected devices ---
    print("\n--- 1. Listing connected devices ---")
    response = requests.get(f"{SERVER_URL}/devices")
    if response.status_code == 200:
        devices = response.json()
        print(f"Found devices: {devices}")
        
        if not devices:
            print("\nNo devices found. Please connect an Android device with USB debugging enabled.")
            return

        device_serial = devices[0]['serial']
        
        # --- 2. Run a shell command ---
        print(f"\n--- 2. Running 'ls /sdcard/' on device {device_serial[:12]}... ---")
        shell_resp = requests.post(f"{SERVER_URL}/{device_serial}/shell", json={"command": "ls /sdcard/"})
        print("Response (first 200 chars):")
        print(shell_resp.json()['output'][:200] + "...")
        
        # --- 3. Take a screenshot ---
        print(f"\n--- 3. Taking a screenshot of device {device_serial[:12]}... ---")
        ss_resp = requests.get(f"{SERVER_URL}/{device_serial}/screenshot")
        if ss_resp.status_code == 200:
            screenshot_path = Path("screenshot.png")
            screenshot_path.write_bytes(ss_resp.content)
            print(f"[SUCCESS] Screenshot saved to '{screenshot_path.resolve()}'")
        else:
            print("[FAILURE] Failed to take screenshot.")
            
    else:
        print(f"Error communicating with server: {response.status_code} {response.text}")


if __name__ == "__main__":
    print("=========================================================")
    print("=== Mobile Integration Server Prototype 📱⚙️ ===")
    print("=========================================================")
    
    if not DEVIN_CORE_AVAILABLE:
        print(f"\nERROR: A core Devin module is missing. Please check your installation. Error: {_import_error}")
    else:
        print("!!! PREREQUISITE: This tool requires an Android device (or emulator) with USB debugging enabled. !!!")
        
        server = None
        try:
            server = MobileIntegrationServer()
            
            # Run the server in a background thread so we can run the client demo
            server_thread = threading.Thread(target=server.run, args=('127.0.0.1', 5006), daemon=True)
            server_thread.start()
            time.sleep(2) # Give the server a moment to start up
            
            run_client_demo()
            
            logger.info("Demo complete. Exiting...")

        except ConnectionError as e:
            logger.critical(f"Could not start server: {e}")
        except Exception as e:
            logger.error(f"An unexpected error occurred: {e}", exc_info=True)


    print("\n=========================================================")
    print("=== Mobile Integration Server Prototype Complete ===")
    print("=========================================================")
