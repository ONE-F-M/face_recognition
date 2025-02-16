import argparse, pickle,time, glob, face_recognition, cv2, json, os, base64, logging, shutil, uuid
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


class Detector:
    def __init__(self, username, bucketpath):
        self.username = username
        self.bucketpath = bucketpath
        Path(f"enroll/images/{self.username}").mkdir(exist_ok=True)
        Path(f"verify/images/{self.username}").mkdir(exist_ok=True)
        os.environ['GOOGLE_APPLICATION_CREDENTIALS'] = 'cred.json'

    def enroll(self, video, filename):
        try:
            generate_images = self.generate_images(video, filename, 'enroll')
            # generate pickle file
            encode_face = self.encode_face()
            return encode_face
        except Exception as e:
            return {'error': True, 'message': str(e)}

    def generate_images(self, video, filename, vtype):
        # reverse the string to video stream
        try:
            if vtype == 'enroll':
                self.IMAGEPATH = PATHCONFIG['enroll']['IMAGEPATH']
                self.VIDEOPATH = PATHCONFIG['enroll']['VIDEOPATH']
                self.ENCODINGPATH = PATHCONFIG['enroll']['ENCODINGPATH']
            else:
                self.IMAGEPATH = PATHCONFIG['verify']['IMAGEPATH']
                self.VIDEOPATH = PATHCONFIG['verify']['VIDEOPATH']
                self.ENCODINGPATH = PATHCONFIG['verify']['ENCODINGPATH']

            video = video.encode('ascii')
            video = base64.b64decode(video)
            video_file = self.VIDEOPATH + f"/" + filename
            with open(video_file, 'wb') as f:
                f.write(video)
            cap = cv2.VideoCapture(video_file)
            success, img = cap.read()
            count = 0
            while success:
                #Resizing the image
                img = cv2.resize(img, (0, 0), fx=0.5, fy=0.5)
                #Limiting the number of images for training. %5 gives 10 images %5.8 -> 8 images %6.7 ->7 images
                if count % 5 == 0:
                    image_file = str(Path(self.IMAGEPATH + f"/{self.username}")) + "/{count}.jpg".format(
                        count=count + 1)
                    cv2.imwrite(image_file, img)
                count = count + 1
                success, img = cap.read()

            # DELETE VIDEO FILE
            if os.path.exists(video_file):
                os.remove(video_file)

            return {'error': False, 'message': 'Image Genrated.'}
        except Exception as e:
            return {'error': True, 'message': str(e)}

    def encode_face(self):
        """
        Loads images in the training directory and builds a dictionary of their
        names and encodings.
        """
        try:
            names = []
            encodings = []

            for filepath in glob.glob(f"{self.IMAGEPATH}/{self.username}/*"):
                filepath = Path(filepath)
                name = filepath.parent.name
                image = face_recognition.load_image_file(filepath)

                face_locations = face_recognition.face_locations(image, model="hog")
                face_encodings = face_recognition.face_encodings(image, face_locations)

                for encoding in face_encodings:
                    names.append(name)
                    encodings.append(encoding)

            name_encodings = {"names": names, "encodings": encodings}
            encoding_file = self.ENCODINGPATH + '/' + self.username + '.pkl'
            with open(encoding_file, mode="wb") as f:
                pickle.dump(name_encodings, f)
            # DELETE TRAINING IMAGES
            if os.path.exists(f"{self.IMAGEPATH}/{self.username}"):
                shutil.rmtree(f"{self.IMAGEPATH}/{self.username}", ignore_errors=True)
            # SEND FILE TO GCP in face_recognition
            try:
                storage_client = storage.Client()
                bucket = storage_client.bucket('face_recognition_v3')
                blobs = storage_client.list_blobs(f'{self.bucketpath}/encoding')
                blob = bucket.blob(f'{self.bucketpath}/encoding/{self.username}.pkl')
                with open(encoding_file, 'rb') as f:
                    blob.upload_from_file(f)
            except Exception as e:
                print(str(e))
            # check if pickle exist in verify
            if os.path.isfile('verify/encoding/' + self.username + '.pkl'):
                os.remove('verify/encoding/' + self.username + '.pkl')
            # manually move the file
            shutil.copyfile('enroll/encoding/' + self.username + '.pkl', 'verify/encoding/' + self.username + '.pkl')
            # DELETE pickle
            if os.path.isfile(self.ENCODINGPATH + '/' + self.username + '.pkl'):
                os.remove(self.ENCODINGPATH + '/' + self.username + '.pkl')
            return {'error': False, 'message': 'success'}
        except Exception as e:
            print(str(e), 'error\n\n')
            return {'error': True, 'message': str(e)}

    def verify(self, video, filename):
        """
        Given an unknown image, get the locations and encodings of any faces and
        compares them against the known encodings to find potential matches.
        """
        try:
            # generate the data
            self.generate_images(video, filename, 'verify')
            # get pickle file
            if not os.path.isfile(self.ENCODINGPATH + f"/{self.username}.pkl"):
                storage_client = storage.Client()
                bucket = storage_client.bucket('face_recognition_v3')
                blob = bucket.blob(f'{self.bucketpath}/encoding/{self.username}.pkl')
                blob.download_to_filename(self.ENCODINGPATH + f"/{self.username}.pkl")
            with Path(self.ENCODINGPATH + f"/{self.username}.pkl").open(mode="rb") as f:
                loaded_encodings = pickle.load(f)

            # start comparing
            found = False
            count = 0
            countT = 0
            countF = 0
            for filepath in glob.glob(f"{self.IMAGEPATH}/{self.username}/*"):
                filepath = Path(filepath)
                input_image = face_recognition.load_image_file(filepath)

                input_face_locations = face_recognition.face_locations(
                    input_image, model="hog"
                )
                input_face_encodings = face_recognition.face_encodings(
                    input_image, input_face_locations
                )
                for bounding_box, unknown_encoding in zip(
                        input_face_locations, input_face_encodings
                ):
                    name = self._recognize_face(unknown_encoding, loaded_encodings)
                    if name:
                        countT += 1
                    count += 1
            # check if matching is >= 50%
            if count == 0:
                return {'error': True, 'message': 'Face not found.', 'text': 'Face not found.'}
            if ((countT / count) * 100) >= 50:
                found = True
            # DELETE IMAGES
            if os.path.exists(f"{self.IMAGEPATH}/{self.username}"):
                shutil.rmtree(f"{self.IMAGEPATH}/{self.username}", ignore_errors=True)
            if not found:
                return {'error': True, 'message': 'We could not verify your face.',
                        'text': 'We could not verify your face.'}
            return {'error': False, 'message': 'Face Verified.'}
        except Exception as e:
            return {'error': True, 'message': str(e)}

    def _recognize_face(self, unknown_encoding, loaded_encodings):
        """
        Given an unknown encoding and all known encodings, find the known
        encoding with the most matches.
        """
        boolean_matches = face_recognition.compare_faces(
            loaded_encodings["encodings"], unknown_encoding
        )
        votes = Counter(
            name
            for match, name in zip(boolean_matches, loaded_encodings["names"])
            if match
        )
        if votes:
            return votes.most_common(1)[0][0]


class AntiSpoof:
    def __init__(self, file_path):
        try:
            current_dir = os.getcwd()
            # Initialize DNN face detector
            self._face_detector = cv2.dnn.readNetFromCaffe(
                "./models/deploy.prototxt",
                "./models/res10_300x300_ssd_iter_140000.caffemodel"
            )
            self._file_path = file_path
            Path(f"verify/anti-spoof/").mkdir(exist_ok=True)
        except:
            logging.error("Exception in detect_liveliness", exc_info=True)
            
    def verify(self):
        # Step 1: Check liveliness
    
        status, message, cap, traceback_info = self.detect_liveliness()
        
        if not status:
            return status, message, traceback_info

        # Step 2: Check blinks
        status, message, traceback_info = self.detect_blinks(cap=cap)
        
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
            
            while True:
                ret, frame = cap.read()
                if not ret:
                    break

                frame_counter += 1
                if frame_counter <= 3:
                    continue  # Skip first 3 frames

                gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                
                # Only detect faces after the first 3 frames
                faces = self.detect_faces_dnn(frame) if frame_counter > 3 else []
                
                if len(faces) == 0:
                    return False, "No face detected!", None, ""
                
                # Detect motion using optical flow
                if prev_gray is not None:
                    motion_score = self.calculate_optical_flow(prev_gray, gray)
                    motion_scores.append(motion_score)

                # Track face position and size
                for (x1, y1, x2, y2) in faces:
                    face_positions.append((x1, y1, x2, y2))

                prev_gray = gray

                # Skip frames for faster processing
                if frame_counter % 3 != 0:
                    continue

                # Ensure liveliness: Check motion and face variability
                if len(motion_scores) > 10:
                    avg_motion = np.mean(motion_scores[-10:])  # Last 10 motion scores
                    if avg_motion < 0.2:  # Threshold for minimal motion
                        return False, "No significant motion detected!", None, ""

                    
                # If liveliness detected
                if len(motion_scores) >= 20 and len(face_positions) >= 20:
                    avg_motion = np.mean(motion_scores[-20:])
                    variances = np.var(face_positions[-20:], axis=0)
                    if avg_motion >= 0.2 and any(variance >= 5 for variance in variances):
                        return True, "", cap, ""

                if frame_counter >= 100:
                    return False, "Whoops! It seems you've triggered our spoof alert radar! Please ensure that your face is moving or check your camera. 🤖", object(), ""

            cap.release()
            cv2.destroyAllWindows()

            return False, "Whoops! It seems you've triggered our spoof alert radar! Please ensure that your face is moving or check your camera. 🤖", object(), ""
        except Exception as e:
            logging.error("Exception in detect_liveliness", exc_info=True)
            return False, str(e), None, format_exc()


    def eye_aspect_ratio(self, eye):
        A = dist.euclidean(eye[1], eye[5])
        B = dist.euclidean(eye[2], eye[4])
        C = dist.euclidean(eye[0], eye[3])
        return (A + B) / (C)

    def adjust_ear_threshold(self, left_ear, right_ear):
        return min(left_ear, right_ear) * 0.8

    def detect_blinks(self, cap, num_blinks_required: int = 2):
        try:
            # Initialize dlib's face detector and the facial landmark predictor
            detector = dlib.get_frontal_face_detector()
            predictor = dlib.shape_predictor("shape_predictor_68_face_landmarks.dat")
            (L_start, L_end) = face_utils.FACIAL_LANDMARKS_IDXS["left_eye"] 
            (R_start, R_end) = face_utils.FACIAL_LANDMARKS_IDXS['right_eye'] 

            # Initialize blink counter
            blink_counter = 0

            # Initialize variables for blink detection
            
            EYE_AR_CONSEC_FRAMES = 3
            COUNTER = 0
            TOTAL = 0
          

            while True:
                ret, frame = cap.read()
                if not ret:
                    break

                gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                rects = detector(gray, 0)
                for rect in rects:
                    shape = predictor(gray, rect)
                    shape = face_utils.shape_to_np(shape)
                    left_eye = shape[L_start: L_end] 
                    right_eye = shape[R_start:R_end] 
                    left_ear = self.eye_aspect_ratio(left_eye)
                    right_ear = self.eye_aspect_ratio(right_eye)

                    ear = (left_ear + right_ear) / 2.0
                    EYE_AR_THRESH = 0.45
                    if ear < EYE_AR_THRESH:
                        COUNTER += 1
                    else:
                        if COUNTER >= EYE_AR_CONSEC_FRAMES:
                            TOTAL += 1
                        else:
                            COUNTER = 0
                
                
                if TOTAL >= num_blinks_required:
                    return True, "", ""

            cap.release()
            cv2.destroyAllWindows()

            return False, "Uh-oh! Blink and you'll miss it! Try blinking a bit more next time. 😉", ""
        except Exception as e:
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
            self.get_path()
            video_path = self.save_video()
            
            status, message, traceback = self.anti_spoof_liveliness(file_path=video_path)
            if not status:
               os.remove(video_path) if os.path.isfile(video_path) else None
               return True, message, traceback

            cap = cv2.VideoCapture(video_path)
            count = 0

            output_dir = Path(self.IMAGEPATH) / f"{self._username}"
            output_dir.mkdir(exist_ok=True)
            img_Count = 1
            while True:
                ret, frame = cap.read()
                
                if not ret:
                    break
                try:
                    face_results = DeepFace.extract_faces(
                        img_path=frame,
                        enforce_detection=False
                    ) 
                except Exception:
                    
                    face_results = []  # in case of an error in detection, skip this frame
                    return True, "An Error Occured while saving Pickle file. Please try again.", f"{format_exc()}"
                for face_dict in face_results:
                    
                    facial_area = face_dict.get("facial_area")
                    if facial_area is None:
                        continue
                    
                    # Optionally resize to a fixed size for consistency.
                    face_image=frame[facial_area.get('y'):facial_area.get('y') + facial_area.get('h'), facial_area.get('x'):facial_area.get('x') + facial_area.get('w')]
                    image_file = output_dir / f"{count + 1}.jpg"
                    face_image = cv2.resize(face_image, (300, 300))
                    cv2.imwrite(str(image_file), face_image)
                    count += 1
                    img_Count+=1
               
            
            cap.release()
            cv2.destroyAllWindows()

            if os.path.exists(video_path):
                os.remove(video_path)

            self.save_faces_to_pickle(images_dir=str(output_dir))
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
        shutil.rmtree(images_dir, ignore_errors=True) if os.path.exists(images_dir) else None

        # check if pickle exist in verify
        os.remove('verify/encoding/' + self._username + '.pkl') if os.path.isfile(
            'verify/encoding/' + self._username + '.pkl') else None

        # manually move the file
        shutil.copyfile('enroll/encoding/' + self._username + '.pkl', 'verify/encoding/' + self._username + '.pkl')

        # DELETE pickle
        os.remove(self.ENCODINGPATH + '/' + self._username + '.pkl') if os.path.isfile(
            self.ENCODINGPATH + '/' + self._username + '.pkl') else None
    
    
    def cosine_similarity(self,emb1, emb2):
        return np.dot(emb1, emb2) / (np.linalg.norm(emb1) * np.linalg.norm(emb2))

    def verify(self):
        try:
            
            self.get_path()
            video_path = self.save_video()

            status, message, traceback = self.anti_spoof_liveliness(file_path=video_path)
            if not status:
               os.remove(video_path) if os.path.isfile(video_path) else None
               return True, message, traceback
            
            
            # Load enrolled faces from pickle file
            if not os.path.isfile(self.ENCODINGPATH + f"/{self._username}.pkl"):
                return True, 'Enrollment Pickle not found', '404 Enrollment Pickle not found'
                # Download pickle file if not available locally
                
            with open(self.ENCODINGPATH + f"/{self._username}.pkl", 'rb') as f:
                enrolled_faces = pickle.load(f)
            # Prepare enrolled faces and labels for training
            user_images = enrolled_faces.get(self._username, [])
            user_faces = []
            img_count=0
            for img in user_images:
                try:
                    # Convert images to embeddings
                    
                    embedding = DeepFace.represent(img, model_name="Facenet", enforce_detection=True)
                    if not embedding:
                        return True, "Error While processing video, Please try again", ""
                    if embedding[0].get('face_confidence')>0.5:
                        embedding = embedding[0]['embedding']
                        user_faces.append(embedding)
                        img_count+=1
                        if img_count>9:
                            break
                except Exception as e:
                    logging.error("Exception in detect_faces_dnn", exc_info=True)
                    return True, "Error processing an image for {self._username}",e
            
            embed_count = 0
            checkin_embeddings = []
            cap = cv2.VideoCapture(video_path)
            detected_faces = [] 
            
            while True:
                ret, frame = cap.read()
                if not ret:
                    break
                if embed_count>9:
                    break
                
               
                detected_faces = DeepFace.represent(frame, model_name="Facenet", enforce_detection=False)
                if not detected_faces:
                    
                    return True, "Error While processing video, Please try again", ""
                if detected_faces[0].get('face_confidence')>0.5:
                    if embed_count>9:
                        break
                    checkin_embedding = detected_faces[0]['embedding']
                    checkin_embeddings.append(checkin_embedding)
                    embed_count+=1

            cap.release()
            cv2.destroyAllWindows()
            match_count = 0
            unmatched_count = 0
            for checkin_emb in checkin_embeddings:
                for stored_emb in user_faces:
                    similarity = self.cosine_similarity(checkin_emb, stored_emb)
                    if similarity > 0.8:
                        match_count += 1
                    else:
                        unmatched_count +=1
            
            
            
            if unmatched_count >match_count:
                return True, "Error 404: Face not recognized.Maybe smile a bit more?", ""

            if match_count > unmatched_count:
                return False, "Face verification Successful", ""

            return True, "Face Verification Failed", ""

        except Exception as e:
            logging.error("Exception in detect_liveliness", exc_info=True)
            return True, str(e), f"{format_exc()}"
    
    @staticmethod
    def anti_spoof_liveliness(file_path: str):
        try:
            anti_spoof = AntiSpoof(file_path=file_path)
            return anti_spoof.verify()
        except Exception as e:
            return False, str(e), f"{format_exc()}"
