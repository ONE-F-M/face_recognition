from face_engine import test_blinks_antispoof
import os
import logging

# Configure logging to see output
logging.basicConfig(level=logging.DEBUG)

# Pick a video
video_path = "/home/frappe/face_recognition/face_recognition/error_cases/mr_s.mp4" # From list
if not os.path.exists(video_path):
    print(f"Video not found: {video_path}")
else:
    print(f"Testing video: {video_path}")
    result = test_blinks_antispoof(video_path)
    print(f"Result: {result}")
