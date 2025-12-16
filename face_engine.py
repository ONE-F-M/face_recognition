import argparse, pickle,time, glob, cv2, json, os, base64, logging, shutil, uuid
from collections import Counter
from deepface import DeepFace
import joblib
from imutils import face_utils 
import logging
from pathlib import Path
from PIL import Image
import numpy as np
from google.cloud import storage
from werkzeug.utils import secure_filename
import dlib
from scipy.spatial import distance as dist
from traceback import format_exc
logging.basicConfig(
                filename="antispoof_errors.log",
                level=logging.DEBUG,
                format="%(asctime)s [%(levelname)s] %(message)s"
            )
# Create directories if they don't already exist
Path("enroll").mkdir(exist_ok=True)
Path("verify").mkdir(exist_ok=True)
Path("error_cases").mkdir(exist_ok=True)
Path("enroll/images").mkdir(exist_ok=True)
Path("enroll/video").mkdir(exist_ok=True)
Path("enroll/encoding").mkdir(exist_ok=True)
Path("verify/images").mkdir(exist_ok=True)
Path("verify/video").mkdir(exist_ok=True)
Path("verify/encoding").mkdir(exist_ok=True)

PATHCONFIG = {
    'enroll': {
        'IMAGEPATH': 'enroll/images',
        'VIDEOPATH': 'enroll/video',
        'ENCODINGPATH': 'enroll/encoding'
    },
    'verify': {
        'IMAGEPATH': 'verify/images',
        'VIDEOPATH': 'verify/video',
        'ENCODINGPATH': 'verify/encoding'
    }
}


def set_credential(data):
    """
        We import the credential by sending it here
    """
    try:
        if os.path.isfile('cred.json'):
            return {'error': False, 'message': 'BigBang, Hello World!'}
        json_object = json.dumps(data['cred'], indent=4)

        # Writing to sample.json
        with open("cred.json", "w") as outfile:
            outfile.write(json_object)

        # get existing pickles
        # download_pickles(data['bucketpath'])
        return {'error': False, 'message': 'BigBang, Hello World!'}
    except Exception as e:
        return {'error': True, 'message': str(e)}


def download_pickles(bucketpath):
    os.environ['GOOGLE_APPLICATION_CREDENTIALS'] = 'cred.json'
    storage_client = storage.Client()
    bucket = storage_client.bucket('face_recognition_v3')
    blobs = bucket.list_blobs(prefix=f'{bucketpath}/encoding/')
    for blob_name in blobs:
        if blob_name.name.endswith('.pkl'):
            blob = bucket.blob(blob_name.name)
            blob.download_to_filename(f"verify/encoding/{blob_name.name.split('/')[-1]}")



class AntiSpoof:
    def __init__(self, file_path,username=None):
        try:
            
            # Initialize DNN face detector
            self._face_detector = cv2.dnn.readNetFromCaffe(
                "./models/deploy.prototxt",
                "./models/res10_300x300_ssd_iter_140000.caffemodel"
            )
            self._file_path = file_path
            self.username = username
            Path(f"verify/anti-spoof/").mkdir(exist_ok=True)
        except:
            logging.error("Exception in Face Recognition", exc_info=True)

    def save_error_video(self):
        try:
            username = self.username if self.username else "unknown"
            error_dir = Path("error_cases") / username
            error_dir.mkdir(parents=True, exist_ok=True)
            
            filename = os.path.basename(self._file_path)
            destination = error_dir / filename
            
            shutil.copy2(self._file_path, destination)
            return str(destination)
        except Exception as e:
            logging.error(f"Failed to save error video: {e}", exc_info=True)
            return None
            
    def verify(self):
        """
        Check Liveliness and detect blinks
        """

        # Step 1: Check blinks and liveliness
        status, message, cap, traceback_info = self.detect_liveliness()
        if not status:
            return status, message, traceback_info
        
        time3 = time.time()
        status, message, traceback_info = self.detect_blinks()
        time4 = time.time()
        
        if not status:
            return status, message, traceback_info

        return True, "", ""
    
    def detect_faces_dnn(self, frame):
        try:
            h, w = frame.shape[:2]

            # Convert to blob for DNN
            blob = cv2.dnn.blobFromImage(
                frame, scalefactor=1.0, size=(300, 300), mean=(104.0, 177.0, 123.0)
            )
            self._face_detector.setInput(blob)

            # Get face detections
            detections = self._face_detector.forward()

            # Parse detections
            faces = []
            
            for i in range(detections.shape[2]):
                confidence = detections[0, 0, i, 2]
                if confidence > 0.5:  # Confidence threshold
                    # Extract bounding box coordinates
                    box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
                    faces.append(box.astype("int"))
            # Return faces
            return faces
        except Exception as e:
            logging.error("Exception in detect_faces_dnn", exc_info=True)
            return []

    def calculate_optical_flow(self, prev_gray, gray):
        flow = cv2.calcOpticalFlowFarneback(
            prev_gray, gray, None, 0.5, 3, 15, 3, 5, 1.2, 0
        )
        mag, _ = cv2.cartToPolar(flow[..., 0], flow[..., 1])
        return mag.mean()

    def detect_liveliness(self):
        try:
            cap = cv2.VideoCapture(self._file_path)
            frame_counter = 0
            prev_gray = None
            motion_scores = []
            face_positions = []
            face_detected_atleast_once = False
            liveliness_detected = False
            last_error = ""
            face_detected_count = 0
            while True:
                ret, frame = cap.read()
                if not ret:
                    break

                frame_counter += 1
                if frame_counter <= 3:
                    continue  # Skip first 3 frames

                gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                if gray.mean() < 80: #Grey Image
                    continue
                    
                # Only detect faces after the first 3 frames
                faces = self.detect_faces_dnn(frame) if frame_counter > 3 else []
                
                
                if len(faces) > 0:
                    face_detected_atleast_once = True
                    
            cap.release()
            cv2.destroyAllWindows()
            if not face_detected_atleast_once:
                self.save_error_video()
                return False, "No face detected!", None, ""

            return True, "", None, ""
        except Exception as e:
            logging.error("Exception in detect_liveliness", exc_info=True)
            self.save_error_video()
            return False, str(e), None, format_exc()


    def eye_aspect_ratio(self, eye):
        A = dist.euclidean(eye[1], eye[5])
        B = dist.euclidean(eye[2], eye[4])
        C = dist.euclidean(eye[0], eye[3])
        return (A + B) / (C)

    def adjust_ear_threshold(self, left_ear, right_ear):
        return min(left_ear, right_ear) * 0.8

    def detect_blinks(self, num_blinks_required: int = 2):
        """
            Detect blinks in the received video
        """
        try:
            # Initialize dlib's face detector and the facial landmark predictor
            
            cap = cv2.VideoCapture(self._file_path)
            detector = dlib.get_frontal_face_detector()
            predictor = dlib.shape_predictor("shape_predictor_68_face_landmarks.dat")
            (L_start, L_end) = face_utils.FACIAL_LANDMARKS_IDXS["left_eye"] 
            (R_start, R_end) = face_utils.FACIAL_LANDMARKS_IDXS['right_eye'] 

            # Initialize blink counter
            blink_counter = 0

            # Initialize variables for blink detection
            
            EYE_AR_CONSEC_FRAMES =  2
            COUNTER = 0
            TOTAL = 0
            rotate_video = False
            # Detect if the video needs rotation
            width = cap.get(cv2.CAP_PROP_FRAME_WIDTH)
            height = cap.get(cv2.CAP_PROP_FRAME_HEIGHT)
            while True:
                
                ret, frame = cap.read()
                
                if not ret:
                    
                    break
                
                gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                
                if gray.mean() < 80:
                    
                    continue
                rects = detector(gray, 0)
                
                for rect in rects:
                    shape = predictor(gray, rect)
                    shape = face_utils.shape_to_np(shape)
                    left_eye = shape[L_start: L_end] 
                    right_eye = shape[R_start:R_end] 
                    left_eye_ratio = self.eye_aspect_ratio(left_eye)
                    right_eye_ratio = self.eye_aspect_ratio(right_eye)
                    # Use logging to view the left, right ear and eye values
                    
                    eye = (left_eye_ratio + right_eye_ratio) / 2.0
                    
                    EYE_AR_THRESH = 0.40
                    
                    if eye < EYE_AR_THRESH:
                        COUNTER += 1
                    else:
                        # The eye is open again
                        if COUNTER >= EYE_AR_CONSEC_FRAMES:
                            # A blink was detected
                            TOTAL += 1
                        # Reset the counter in both cases
                        COUNTER = 0
                
                    
               
                # If required blinks detected
                if TOTAL >= num_blinks_required:
                    return True, "", ""
            cap.release()
            cv2.destroyAllWindows()
            self.save_error_video()
            return False, "Uh-oh! Blink and you'll miss it! Try blinking a bit more next time. 😉", ""
        except Exception as e:
            self.save_error_video()
            return False, str(e), f"{format_exc()}"





class FaceRecognition:

    def __init__(self, username: str, the_type: str, filename: str, video, decrypt_video: int) -> None:
        self._bucketpath = os.getenv("BUCKETPATH", "face_recognition_v3/testing/encoding")
        self._video = video
        self._decrypt_video = int(decrypt_video)
        self._username = username
        self._type = the_type
        self._filename = filename
        self._face_detector = cv2.dnn.readNetFromCaffe(
                "./models/deploy.prototxt",
                "./models/res10_300x300_ssd_iter_140000.caffemodel"
            )
        self._face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
        Path(f"enroll/images/{self._username}").mkdir(exist_ok=True)
        Path(f"verify/images/{self._username}").mkdir(exist_ok=True)
        os.environ['GOOGLE_APPLICATION_CREDENTIALS'] = "cred.json"

    def get_path(self) -> None:
        if self._type == 'enroll':
            self.IMAGEPATH = PATHCONFIG['enroll']['IMAGEPATH']
            self.VIDEOPATH = PATHCONFIG['enroll']['VIDEOPATH']
            self.ENCODINGPATH = PATHCONFIG['enroll']['ENCODINGPATH']
        else:
            self.IMAGEPATH = PATHCONFIG['verify']['IMAGEPATH']
            self.VIDEOPATH = PATHCONFIG['verify']['VIDEOPATH']
            self.ENCODINGPATH = PATHCONFIG['verify']['ENCODINGPATH']

    def save_video(self) -> str:
        video_path = self.VIDEOPATH + f"/" + self._filename
        with open(video_path, 'wb') as f:
            if self._decrypt_video:
                f.write(base64.b64decode(self._video.read()))
            else:
                f.write(self._video.read())
            # f.write(base64.b64decode(self._video.read()))

        return video_path

    def detect_faces(self, image):
        # Convert the image to grayscale
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

        # Detect faces in the image
        faces = self._face_cascade.detectMultiScale(gray, scaleFactor=1.3, minNeighbors=5)

        return faces

    def detect_faces_dnn(self, frame):
        try:
            h, w = frame.shape[:2]

            # Convert to blob for DNN
            blob = cv2.dnn.blobFromImage(
                frame, scalefactor=1.0, size=(300, 300), mean=(104.0, 177.0, 123.0)
            )
            self._face_detector.setInput(blob)

            # Get face detections
            detections = self._face_detector.forward()

            # Parse detections
            faces = []
            for i in range(detections.shape[2]):
                confidence = detections[0, 0, i, 2]
                if confidence > 0.5:  # Confidence threshold
                    # Extract bounding box coordinates
                    box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
                    faces.append(box.astype("int"))
            # Return faces
            return faces
        except Exception as e:
            logging.error("Exception in detect_faces_dnn", exc_info=True)
            return []

    def enroll(self) -> tuple:
        try:
            logging.debug("Starting Enrollment Process")
            self.get_path()
            video_path = self.save_video()
            
            status, message, traceback = self.anti_spoof_liveliness(file_path=video_path,username=self._username)
            if not status:
               os.remove(video_path) if os.path.isfile(video_path) else None
               return True, message, traceback

            cap = cv2.VideoCapture(video_path)
            framecount = 0
            writecount = 0
            # shutil.rmtree(f"{self.IMAGEPATH}/{self._username}", ignore_errors=True)
            output_dir = Path(self.IMAGEPATH) / f"{self._username}"
            output_dir.mkdir(exist_ok=True)
            rotate_video = False
            
            while True:
                try:
                    ret, frame = cap.read()
                    if not ret:
                        break
                    
                    faces = self.detect_faces_dnn(frame)
                    if len(faces) == 0:
                        continue
                    else:
                        
                        if writecount < 5: #saving 5 images to reference vs the checkin videos, skipping the first 4 because of dark images
                        # Optionally resize to a fixed size for consistency.
                            image_file = self.IMAGEPATH+ '/' +f"{framecount + 1}.jpg"
                            image_file = str(output_dir) + "/{count}.jpg".format(count=framecount + 1)
                            face_image = cv2.resize(frame, (300, 300))
                            cv2.imwrite(str(image_file), face_image)
                            writecount += 1
                        else:
                            break  
                        framecount += 1
                except Exception:
                    logging.error("Exception in detect_liveliness", exc_info=True)
            
            cap.release()
            cv2.destroyAllWindows()
            if os.path.exists(video_path):
                os.remove(video_path)
            return False, "Enrollment Successful", ""
        except Exception as e:
            return True, str(e), f"{format_exc()}"


    def save_faces_to_pickle(self, images_dir):
        faces = {self._username: list()}
        for filename in os.listdir(images_dir):
            if filename.endswith(".jpg"):
                img_path = os.path.join(images_dir, filename)
                face_img = cv2.imread(img_path)
                faces[self._username].append(face_img)

        pickle_file_path = self.ENCODINGPATH + '/' + self._username + '.pkl'
        with open(pickle_file_path, 'wb') as new_file:
            pickle.dump(faces, new_file)

        # DELETE TRAINING IMAGES
        # shutil.rmtree(images_dir, ignore_errors=True) if os.path.exists(images_dir) else None

        # check if pickle exist in verify
        os.remove('verify/encoding/' + self._username + '.pkl') if os.path.isfile(
            'verify/encoding/' + self._username + '.pkl') else None

        # manually move the file
        shutil.copyfile('enroll/encoding/' + self._username + '.pkl', 'verify/encoding/' + self._username + '.pkl')

        # DELETE pickle
        os.remove(self.ENCODINGPATH + '/' + self._username + '.pkl') if os.path.isfile(
            self.ENCODINGPATH + '/' + self._username + '.pkl') else None
    
    
   
    
    def verify(self):
        try:
            """
                Verify that the video received passes the anti spoof liveliness check
            """
            self.get_path()
            video_path = self.save_video()
            time1 = time.time()
            status, message, traceback = self.anti_spoof_liveliness(file_path=video_path,username=self._username)
            time2 = time.time()
            logging.debug(f"Liveliness Time Taken : {time2 - time1} seconds")
            if not status:
            #    os.remove(video_path) if os.path.isfile(video_path) else None
               return True, message, traceback
            
            
            # Load enrolled faces from pickle file
            
            if not os.path.isfile('enroll'+'/images'+ f"/{self._username}"+ "/1.jpg"):
                return True, 'Enrollment Images not found', '404 Enrollment Images not found.'
                # Download pickle file if not available locally
                
            
            write_count=0
            
            enrollment_image_folder = "enroll/images"+f"/{self._username}"
            checkin_image_folder = "verify/images"+f"/{self._username}"
            
            Path(checkin_image_folder).mkdir(exist_ok=True)
            cap = cv2.VideoCapture(video_path)
            time3 = time.time()
            while True:
                if write_count>10:
                    break
                ret, frame = cap.read()

                if not ret:
                    break
                
                faces = self.detect_faces_dnn(frame)
                if len(faces) == 0:
                    continue
                else:
                    if write_count>3 and write_count < 9:
                        image_file = checkin_image_folder+ '/' +f"{write_count + 1}.jpg"
                        face_image = cv2.resize(frame, (300, 300))
                        cv2.imwrite(str(image_file), face_image)
                        
                    write_count += 1
            cap.release()
            cv2.destroyAllWindows()
            match_count = 0
            unmatched_count = 0
            checkin_images = [os.path.join(checkin_image_folder, img) for img in os.listdir(checkin_image_folder) if img.lower().endswith(('.jpg', '.jpeg', '.png'))]
            enrollment_images = [os.path.join(enrollment_image_folder, img) for img in os.listdir(enrollment_image_folder) if img.lower().endswith(('.jpg', '.jpeg', '.png'))]
            confidence_threshold = 5
            for checkin_image in checkin_images:
                if match_count > 8 or unmatched_count > 8 or abs(match_count - unmatched_count) >= confidence_threshold:
                    break          
                for enrollment_image in enrollment_images:
                    if match_count > 8 or unmatched_count > 8 or abs(match_count - unmatched_count) >= confidence_threshold:
                        break
                    try:
                        result = auto_threshold_verify(checkin_image,enrollment_image,model_name ="Dlib",distance_metric="euclidean",detector_backend="dlib")
                        if result.get('verified'):
                            match_count+=1
                        else:
                            unmatched_count+=1
                    except Exception as e:
                        logging.error("Exception in Verification", exc_info=True)
                
            # Iterate over each image in folder Fc
            time4 = time.time()
            logging.debug(f"Verification Time Taken : {time4 - time3} seconds")
            os.remove(video_path) if os.path.isfile(video_path) else None
            shutil.rmtree(checkin_image_folder, ignore_errors=True) if os.path.exists(checkin_image_folder) else None
            
            if match_count >= unmatched_count:
                return False, "Face verification Successful", ""
            else:
                 return True, "Error 404: Face not recognized.Maybe smile a bit more?", ""
            
            

        except Exception as e:
            logging.error("Exception while checking", exc_info=True)
            return True, str(e), f"{format_exc()}"
    
    @staticmethod
    def anti_spoof_liveliness(file_path: str,username: str):
        try:
            anti_spoof = AntiSpoof(file_path=file_path,username=username)
            return anti_spoof.verify()
        except Exception as e:
            return False, str(e), f"{format_exc()}"

def auto_threshold_verify(img1_path, img2_path, model_name="Dlib", detector_backend=None, distance_metric="euclidean"):
    # Base threshold for verification
    # Perform verification
    if not detector_backend:
        result = DeepFace.verify(img1_path, img2_path, model_name=model_name, distance_metric=distance_metric,enforce_detection=False)
    else:
        result = DeepFace.verify(img1_path, img2_path, model_name=model_name, detector_backend=detector_backend, distance_metric=distance_metric,enforce_detection=False)
    
    if not result['verified']:
        base_threshold = float(result.get('threshold'))
        if not base_threshold:
            return result
        # Extract the distance
        distance = float(result['distance'])
        adjusted_threshold = float(base_threshold+0.10)
        # Adjust threshold automatically
        if distance > adjusted_threshold:
            return result
        # Set verification status based on adjusted threshold
        result['verified'] = distance <= adjusted_threshold
        result['adjusted_threshold'] = adjusted_threshold

    return result

    
def verify_for_user(user_name):
    enrollment_images_folder = 'enroll'+'/images'+ f"/{user_name}"
    checkin_images_folder = 'verify'+'/images'+ f"/{user_name}"
    for each in [enrollment_images_folder,checkin_images_folder]:
        jpg_files = [f for f in os.listdir(each) if f.lower().endswith(".jpg")]
        if not jpg_files:
            print(f"No Images found in {each}")
            return
    match_count = 0
    unmatched_count = 0
    match_count_1 = 0
    unmatched_count_1 = 0
    match_count_2 = 0
    unmatched_count_2 = 0
    match_count_4 = 0
    unmatched_count_4 = 0
    match_count_3 = 0
    unmatched_count_3 = 0
    checkin_images = [os.path.join(checkin_images_folder, img) for img in os.listdir(checkin_images_folder) if img.lower().endswith(('.jpg', '.jpeg', '.png'))]
    enrollment_images = [os.path.join(enrollment_images_folder, img) for img in os.listdir(enrollment_images_folder) if img.lower().endswith(('.jpg', '.jpeg', '.png'))]
    
    for checkin_image in checkin_images:
        if match_count>9:
            break 
        if unmatched_count>9:
            break                
        for enrollment_image in enrollment_images:
            if match_count>9:
                break
            if unmatched_count>9:
                break
            try:
                result = auto_threshold_verify(checkin_image,enrollment_image,model_name ="Dlib",distance_metric="euclidean")
                
                result1 = auto_threshold_verify(checkin_image,enrollment_image,model_name ="Dlib",distance_metric="cosine")
                
                result2 = auto_threshold_verify(checkin_image,enrollment_image,model_name ="Dlib",distance_metric="euclidean",detector_backend="dlib")
               
                result3 = auto_threshold_verify(checkin_image,enrollment_image,model_name ="Dlib",distance_metric="euclidean",detector_backend="mtcnn")
                
                result4 = auto_threshold_verify(checkin_image,enrollment_image,model_name ="Dlib",detector_backend="mtcnn")
                
               
                if result.get('verified'):
                    match_count+=1
                else:
                    unmatched_count+=1
                if result1.get('verified'):
                    match_count_1+=1
                else:
                    unmatched_count_1+=1
                if result2.get('verified'):
                    match_count_2+=1
                else:
                    unmatched_count_2+=1
                if result3.get('verified'):
                    match_count_3+=1
                else:
                    unmatched_count_3+=1
                if result4.get('verified'):
                    match_count_4+=1
                else:
                    unmatched_count_4+=1
            except Exception as e:
                logging.error("Exception in Verification", exc_info=True)
    logging.debug(f"Checkin Results for Result : MATCH COUNT: {match_count} UNMATCHED COUNT: {unmatched_count}")
    logging.debug(f"Checkin Results for Result 1: MATCH COUNT: {match_count_1} UNMATCHED COUNT: {unmatched_count_1}")
    logging.debug(f"Checkin Results for Result 2: MATCH COUNT: {match_count_2} UNMATCHED COUNT: {unmatched_count_2}")
    logging.debug(f"Checkin Results for Result 3: MATCH COUNT: {match_count_3} UNMATCHED COUNT: {unmatched_count_3}")
    logging.debug(f"Checkin Results: for Result 4 MATCH COUNT : {match_count_4} UNMATCHED COUNT: {unmatched_count_4}")

