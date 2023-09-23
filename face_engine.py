import argparse, pickle, glob, face_recognition, cv2, json, os, base64, logging, shutil, dlib
from scipy.spatial import distance as dist
from collections import Counter
from pathlib import Path
from PIL import Image
import numpy as np
from google.cloud import storage

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
        # os.environ['GOOGLE_APPLICATION_CREDENTIALS'] = 'cred.json'

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
            print("\n\n\n\n")
            print("SUCCESS 01")
            print(success)
            print("\n\n\n\n")
            
            while success:
                is_face_live = live_face_detector.detect_live_face(img)
                
                # if is_face_live:
                if 1:
                    # Resizing the image
                    img = cv2.resize(img, (0, 0), fx=0.5, fy=0.5)
                    print("\n\n\n\n")
                    print(video_file)
                    print("SUCCESS O")
                    print("\n\n\n\n")
                    # Limiting the number of images for training. %5 gives 10 images %5.8 -> 8 images %6.7 ->7 images
                    if count % 5 == 0:
                        image_file = str(Path(self.IMAGEPATH + f"/{self.username}")) + "/{count}.jpg".format(
                            count=count + 1)
                        cv2.imwrite(image_file, img)
                        print("\n\n\n\n")
                        print('WRITE FILE')
                        print(image_file)
                        print("\n\n\n\n")
                count = count + 1
                success, img = cap.read()

            # DELETE VIDEO FILE
            if os.path.exists(video_file):
                os.remove(video_file)

            return {'error': False, 'message': 'Image Generated.'}
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
                print('\n\n\n')
                print("ENROLLED ENCODING")
                print(name_encodings)
                print(encoding_file)
                print('\n\n\n')
            # DELETE TRAINING IMAGES
            # if os.path.exists(f"{self.IMAGEPATH}/{self.username}"):
            #     shutil.rmtree(f"{self.IMAGEPATH}/{self.username}", ignore_errors=True)
            # SEND FILE TO GCP in face_recognition
            # try:
            #     storage_client = storage.Client()
            #     bucket = storage_client.bucket('face_recognition_v3')
            #     blobs = storage_client.list_blobs(f'{self.bucketpath}/encoding')
            #     blob = bucket.blob(f'{self.bucketpath}/encoding/{self.username}.pkl')
            #     with open(encoding_file, 'rb') as f:
            #         blob.upload_from_file(f)
            # except Exception as e:
            #     print(str(e))
            # check if pickle exist in verify
            if os.path.isfile('verify/encoding/' + self.username + '.pkl'):
                os.remove('verify/encoding/' + self.username + '.pkl')
            # manually move the file
            shutil.copyfile('enroll/encoding/' + self.username + '.pkl', 'verify/encoding/' + self.username + '.pkl')
            # DELETE pickle
            # if os.path.isfile(self.ENCODINGPATH + '/' + self.username + '.pkl'):
            #     os.remove(self.ENCODINGPATH + '/' + self.username + '.pkl')
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
            # if not os.path.isfile(self.ENCODINGPATH + f"/{self.username}.pkl"):
            #     storage_client = storage.Client()
            #     bucket = storage_client.bucket('face_recognition_v3')
            #     blob = bucket.blob(f'{self.bucketpath}/encoding/{self.username}.pkl')
            #     blob.download_to_filename(self.ENCODINGPATH + f"/{self.username}.pkl")
            with Path(PATHCONFIG['enroll']['ENCODINGPATH'] + f"/{self.username}.pkl").open(mode="rb") as f:
                loaded_encodings = pickle.load(f)
            print("\n\n\n")
            print("LOADED ENCODEINGS")
            print(self.ENCODINGPATH)
            print(PATHCONFIG['enroll']['ENCODINGPATH'])
            print(PATHCONFIG['enroll']['ENCODINGPATH'] + f"/{self.username}.pk")
            print(os.getcwd())
            print(loaded_encodings)
            print("\n\n\n")
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


class LiveFaceDetector:
    def __init__(self):
        # Initialize face detector and facial landmarks predictor
        self.face_detector = dlib.get_frontal_face_detector()
        self.landmark_predictor = dlib.shape_predictor("shape_predictor_68_face_landmarks.dat")

        # Initialize blink detection parameters
        self.EYE_AR_THRESH = 0.25  # Eye aspect ratio (EAR) threshold for blink detection
        self.EYE_AR_CONSEC_FRAMES = 3  # Number of consecutive frames for a blink to be detected
        self.left_eye_start, self.left_eye_end = 42, 48
        self.right_eye_start, self.right_eye_end = 36, 42

        # Initialize blink counters
        self.left_eye_counter = 0
        self.right_eye_counter = 0
        self.total_blinks = 0

    def eye_aspect_ratio(self, eye):
        # Compute the euclidean distances between the two sets of vertical eye landmarks (x, y)-coordinates
        A = dist.euclidean(eye[1], eye[5])
        B = dist.euclidean(eye[2], eye[4])

        # Compute the euclidean distance between the horizontal eye landmark (x, y)-coordinates
        C = dist.euclidean(eye[0], eye[3])

        # Compute the eye aspect ratio (EAR)
        ear = (A + B) / (2.0 * C)

        return ear

    def detect_live_face(self, frame):
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = self.face_detector(gray)
        
        for face in faces:
            # Detect facial landmarks
            landmarks = self.landmark_predictor(gray, face)
            landmarks = dlib.shape_to_np(landmarks)

            # Extract left and right eye landmarks
            left_eye = landmarks[self.left_eye_start:self.left_eye_end]
            right_eye = landmarks[self.right_eye_start:self.right_eye_end]

            # Calculate eye aspect ratios (EAR)
            left_ear = self.eye_aspect_ratio(left_eye)
            right_ear = self.eye_aspect_ratio(right_eye)

            # Average the EAR of both eyes
            avg_ear = (left_ear + right_ear) / 2.0

            # Check for blink
            if avg_ear < self.EYE_AR_THRESH:
                
                if self.left_eye_counter >= self.EYE_AR_CONSEC_FRAMES and self.right_eye_counter >= self.EYE_AR_CONSEC_FRAMES:
                    self.total_blinks += 1
                self.left_eye_counter = 0
                self.right_eye_counter = 0
            else:
                
                self.left_eye_counter += 1
                self.right_eye_counter += 1

        # Check if there are blinks in consecutive frames
        if self.total_blinks > 0:
            return True  # Face is live
        else:
            return False  # Face is not live


# Initialize the LiveFaceDetector
live_face_detector = LiveFaceDetector()


