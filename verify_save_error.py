
import sys
from unittest.mock import MagicMock
import os
from pathlib import Path
import shutil

# Mock dependencies
sys.modules["deepface"] = MagicMock()
sys.modules["imutils"] = MagicMock()
sys.modules["imutils.face_utils"] = MagicMock()
sys.modules["dlib"] = MagicMock()
sys.modules["google.cloud"] = MagicMock()
sys.modules["werkzeug.utils"] = MagicMock()
sys.modules["scipy.spatial"] = MagicMock()

# Mock cv2
cv2_mock = MagicMock()
sys.modules["cv2"] = cv2_mock

# Now import the class
# We need to ensure we can import face_engine even if it has other top-level code that fails
# checking face_engine.py again... 

# face_engine.py imports:
# import argparse, pickle,time, glob, cv2, json, os, base64, logging, shutil, uuid
# from collections import Counter
# from deepface import DeepFace  <-- Mocked
# import joblib <-- Might need mocking
# from imutils import face_utils <-- Mocked
# import logging
# from pathlib import Path
# from PIL import Image <-- Might need mocking
# import numpy as np
# from google.cloud import storage <-- Mocked
# from werkzeug.utils import secure_filename <-- Mocked
# import dlib <-- Mocked
# from scipy.spatial import distance as dist <-- Mocked
# from traceback import format_exc

# Also need to mock joblib, PIL, numpy
sys.modules["joblib"] = MagicMock()
sys.modules["PIL"] = MagicMock()
sys.modules["numpy"] = MagicMock()

# Import
try:
    from face_engine import AntiSpoof
except ImportError as e:
    print(f"ImportError during mock setup: {e}")
    sys.exit(1)
except Exception as e:
    print(f"Exception during import: {e}")
    # It might fail in Path("enroll").mkdir(exist_ok=True) if permission denied? 
    # But face_engine.py has top level Path creation.
    # The user has write access to the directory, so that should be fine.
    # Let's see.
    sys.exit(1)

def test_save_error_video():
    # Setup
    test_video_path = "test_video.mp4"
    with open(test_video_path, 'wb') as f:
        f.write(b'dummy content')
    
    username = "testuser"
    
    # We need to mock cv2.dnn.readNetFromCaffe called in __init__
    # instance = AntiSpoof(...)
    # self._face_detector = cv2.dnn.readNetFromCaffe(...)
    
    # Since we mocked cv2, cv2.dnn.readNetFromCaffe is already a mock and won't fail.
    
    antispoof = AntiSpoof(test_video_path, username=username)
    
    # Execution
    saved_path = antispoof.save_error_video()
    
    # Verification
    expected_path = Path("error_cases") / username / test_video_path
    
    if saved_path and os.path.exists(saved_path) and Path(saved_path) == expected_path:
        print("SUCCESS: Video saved correctly to", saved_path)
    else:
        print(f"FAILURE: Video not saved correctly. Expected {expected_path}, got {saved_path}")

    # Cleanup
    if os.path.exists(test_video_path):
        os.remove(test_video_path)
    if os.path.exists("error_cases"):
        shutil.rmtree("error_cases")

if __name__ == "__main__":
    test_save_error_video()
