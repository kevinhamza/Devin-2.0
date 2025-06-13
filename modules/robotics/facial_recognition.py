# Devin/modules/robotics/facial_recognition.py
# Purpose: Provides facial recognition capabilities, including finding faces in
#          images and identifying them against a database of known individuals.

import logging
import os
from dataclasses import dataclass
from typing import List, Tuple, Dict, Optional
import pickle

# --- Dependency Installation Notes ---
# This module requires several powerful libraries. Installation can be complex.
#
# 1. OpenCV: For image processing.
#    pip install opencv-python
#
# 2. NumPy: For numerical operations.
#    pip install numpy
#
# 3. face_recognition: A high-level library for this task.
#    This library depends on dlib.
#    On Linux/macOS:
#       pip install dlib
#       pip install face_recognition
#    On Windows, installing dlib can be tricky. It often requires installing
#    CMake and a C++ compiler (like Visual Studio Build Tools).
#

try:
    import cv2
    import numpy as np
    import face_recognition
    CV_LIBS_AVAILABLE = True
except ImportError:
    CV_LIBS_AVAILABLE = False
    cv2, np, face_recognition = None, None, None
    logger.error("Required libraries not found! Please install 'opencv-python', 'numpy', and 'face_recognition'. This module will be non-functional.")

# Configure basic logging
logger = logging.getLogger("FacialRecognition")
if not logger.handlers:
    _console_handler = logging.StreamHandler()
    _console_handler.setFormatter(logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s'))
    logger.addHandler(_console_handler)
    logger.setLevel(logging.INFO)

@dataclass
class Recognition:
    """Represents a single recognized face in an image."""
    name: str  # The name of the person, or "Unknown"
    # Bounding box in CSS order (top, right, bottom, left)
    location: Tuple[int, int, int, int]

class FacialRecognizer:
    """
    Recognizes faces in images by comparing them to a pre-encoded
    database of known faces.
    """
    def __init__(self, known_faces_dir: str = "known_faces"):
        """
        Initializes the recognizer by loading and encoding faces
        from a specified directory.
        """
        if not CV_LIBS_AVAILABLE:
            self.known_encodings = []
            self.known_names = []
            logger.error("FacialRecognizer could not be initialized due to missing libraries.")
            return

        self.known_faces_dir = known_faces_dir
        self.known_encodings: List[np.ndarray] = []
        self.known_names: List[str] = []
        
        logger.info("Initializing FacialRecognizer...")
        self._load_or_create_known_faces()

    def _load_or_create_known_faces(self):
        """

        Loads known face encodings from a file if it exists, otherwise
        creates them by scanning the known_faces_dir. This saves time on startup.
        """
        encodings_file = os.path.join(self.known_faces_dir, "encodings.pkl")
        try:
            if os.path.exists(encodings_file):
                logger.info(f"Loading known face encodings from '{encodings_file}'...")
                with open(encodings_file, 'rb') as f:
                    data = pickle.load(f)
                    self.known_encodings = data['encodings']
                    self.known_names = data['names']
                logger.info(f"Loaded {len(self.known_names)} known faces.")
                return
        except Exception as e:
            logger.warning(f"Could not load encodings file: {e}. Re-encoding from images.")

        logger.info(f"No pre-computed encodings found. Scanning '{self.known_faces_dir}' for images...")
        if not os.path.isdir(self.known_faces_dir):
            logger.warning(f"Directory '{self.known_faces_dir}' not found. No known faces will be loaded.")
            return

        for filename in os.listdir(self.known_faces_dir):
            if filename.lower().endswith((".png", ".jpg", ".jpeg")):
                name = os.path.splitext(filename)[0].replace("_", " ").title()
                image_path = os.path.join(self.known_faces_dir, filename)
                
                logger.info(f"  Processing image for '{name}' from '{filename}'...")
                image = face_recognition.load_image_file(image_path)
                # Get encodings. An image might have multiple faces, but we assume one per file for training.
                encodings = face_recognition.face_encodings(image)
                
                if encodings:
                    self.known_encodings.append(encodings[0]) # Add the first face encoding found
                    self.known_names.append(name)
        
        if self.known_encodings:
             logger.info(f"Saving {len(self.known_names)} new encodings to '{encodings_file}' for faster startup next time.")
             with open(encodings_file, 'wb') as f:
                 pickle.dump({"encodings": self.known_encodings, "names": self.known_names}, f)

    def find_and_recognize_faces(self, image_bgr: np.ndarray, tolerance: float = 0.6) -> List[Recognition]:
        """
        Finds all faces in an image and identifies them.

        Args:
            image_bgr (np.ndarray): The input image in OpenCV format (BGR).
            tolerance (float): How much distance between faces to consider it a match.
                               Lower is more strict. 0.6 is a good default.

        Returns:
            List[Recognition]: A list of recognized faces.
        """
        if not self.known_encodings:
            logger.warning("No known faces loaded. All faces will be 'Unknown'.")
        
        # Convert the image from BGR (OpenCV default) to RGB (face_recognition default)
        image_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)

        # 1. Find all face locations and create their encodings
        logger.info("Detecting face locations and computing encodings...")
        face_locations = face_recognition.face_locations(image_rgb)
        face_encodings = face_recognition.face_encodings(image_rgb, face_locations)

        recognitions = []
        for location, encoding in zip(face_locations, face_encodings):
            # 2. Compare the new face encoding with all known face encodings
            matches = face_recognition.compare_faces(self.known_encodings, encoding, tolerance=tolerance)
            name = "Unknown"

            # 3. Find the best match if one exists
            if True in matches:
                face_distances = face_recognition.face_distance(self.known_encodings, encoding)
                best_match_index = np.argmin(face_distances)
                if matches[best_match_index]:
                    name = self.known_names[best_match_index]
            
            recognitions.append(Recognition(name=name, location=location))
            logger.info(f"  Found face: {name} at location {location}")
            
        return recognitions

    @staticmethod
    def draw_recognitions(image: np.ndarray, recognitions: List[Recognition]) -> np.ndarray:
        """Draws bounding boxes and names on an image."""
        img_copy = image.copy()
        for recog in recognitions:
            top, right, bottom, left = recog.location
            name = recog.name
            
            # Draw a box around the face
            color = (0, 255, 0) if name != "Unknown" else (0, 0, 255)
            cv2.rectangle(img_copy, (left, top), (right, bottom), color, 2)
            
            # Draw a label with a name below the face
            cv2.rectangle(img_copy, (left, bottom - 35), (right, bottom), color, cv2.FILLED)
            font = cv2.FONT_HERSHEY_DUPLEX
            cv2.putText(img_copy, name, (left + 6, bottom - 6), font, 1.0, (255, 255, 255), 1)
        return img_copy

# --- Example Usage ---
if __name__ == "__main__":
    print("=========================================================")
    print("=== Robotics Facial Recognition Prototype 👤👁️ ===")
    print("=========================================================")

    if not CV_LIBS_AVAILABLE:
        print("\nRequired libraries not found. Please see installation notes in the script.")
    else:
        # --- 1. Setup the environment for the demo ---
        # Create a directory for known faces and add some dummy images
        KNOWN_FACES_DIR = "known_faces_demo"
        if not os.path.exists(KNOWN_FACES_DIR):
            os.makedirs(KNOWN_FACES_DIR)
        
        # NOTE: For this demo to work, you need to place images of people
        # in the 'known_faces_demo' directory. Name the files like 'Person_Name.jpg'.
        # For now, we will create dummy placeholders.
        print(f"Please place JPG or PNG images of faces to recognize in the '{KNOWN_FACES_DIR}' directory.")
        print("For example: 'Elon_Musk.jpg', 'Marie_Curie.png'")
        
        # Create a dummy test image
        TEST_IMAGE_PATH = "test_recognition_image.jpg"
        dummy_image = np.zeros((600, 800, 3), dtype="uint8")
        cv2.putText(dummy_image, "Place a test image here", (50, 300), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
        cv2.imwrite(TEST_IMAGE_PATH, dummy_image)
        print(f"A placeholder test image has been saved to '{TEST_IMAGE_PATH}'. Please replace it with an image containing faces.")
        
        # --- 2. Initialize the Recognizer ---
        # This will scan the directory and create encodings.
        recognizer = FacialRecognizer(known_faces_dir=KNOWN_FACES_DIR)
        
        # --- 3. Load test image and perform recognition ---
        if recognizer.known_names:
            print(f"\nLoading test image from '{TEST_IMAGE_PATH}'...")
            image_to_test = cv2.imread(TEST_IMAGE_PATH)
            
            if image_to_test is not None:
                recognitions = recognizer.find_and_recognize_faces(image_to_test)
                
                # --- 4. Print results and save output image ---
                print("\n--- Recognition Results ---")
                if recognitions:
                    for recog in recognitions:
                        print(f"  - Found '{recog.name}'")
                    
                    output_image = recognizer.draw_recognitions(image_to_test, recognitions)
                    OUTPUT_IMAGE_PATH = "recognition_output.jpg"
                    cv2.imwrite(OUTPUT_IMAGE_PATH, output_image)
                    print(f"\nOutput image with recognitions saved to '{OUTPUT_IMAGE_PATH}'")
                else:
                    print("  No faces were detected in the test image.")
            else:
                print(f"  Could not load the test image at '{TEST_IMAGE_PATH}'.")
        else:
            print("\nSkipping recognition demo because no known faces were loaded.")

    print("\n=========================================================")
    print("=== Facial Recognition Prototype Complete ===")
    print("=========================================================")
