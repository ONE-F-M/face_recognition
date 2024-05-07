import argparse, pickle, glob, face_recognition, cv2, json, os, base64, logging, shutil, uuid
from collections import Counter
import joblib
from pathlib import Path
from PIL import Image
import numpy as np
from google.cloud import storage
from werkzeug.utils import secure_filename
import dlib
from scipy.spatial import distance as dist

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
        self._face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
        self._file_path = file_path
        Path(f"verify/anti-spoof/").mkdir(exist_ok=True)
    
    
    def verify(self):
        status, message, cap = self.detect_liveliness()
        if not status:
            return status, message
        
        status, message = self.detect_blinks(cap=cap)
        if not status:
            return status, message
        
        return True, ""
        
            
    def detect_liveliness(self):
        # Initialize video capture
        
        cap = cv2.VideoCapture(self._file_path)

        # Initialize frame counter
        frame_counter = 0

        # Initialize variables for motion detection
        prev_frame = None

        while True:
            ret, frame = cap.read()
            if not ret:
                break

            # Convert frame to grayscale for face detection
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

            # Detect faces in the frame
            faces = self._face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))

            # Check if faces are detected
            if len(faces) < 1:
                # Face detected, liveliness check passed
                return False, "Oops! We could not detect a real face. Looks like your face decided to play hide and seek with the camera! 🙈", object()

            # Check for motion
            if prev_frame is not None:
                diff_frame = cv2.absdiff(prev_frame, gray)
                _, thresh_frame = cv2.threshold(diff_frame, 20, 255, cv2.THRESH_BINARY)
                contours, _ = cv2.findContours(thresh_frame.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

                for contour in contours:
                    if cv2.contourArea(contour) > 100:
                        return True, "", cap

            # Update previous frame
            prev_frame = gray

            # Increment frame counter
            frame_counter += 1

            # Break loop if enough frames have been analyzed
            if frame_counter >= 100:
                return False, "Whoops! It seems you've triggered our spoof alert radar! Please ensure that your face is moving or check your camera. 🤖", object()
            
        # Release video capture
        cap.release()
        cv2.destroyAllWindows()

        return False, "Whoops! It seems you've triggered our spoof alert radar! Please ensure that your face is moving or check your camera. 🤖", object()
    
    
    def eye_aspect_ratio(self, eye):
        # Compute the euclidean distances between the two sets of
        # vertical eye landmarks (x, y)-coordinates
        A = dist.euclidean(eye[1], eye[5])
        B = dist.euclidean(eye[2], eye[4])

        # Compute the euclidean distance between the horizontal
        # eye landmark (x, y)-coordinates
        C = dist.euclidean(eye[0], eye[3])

        # Compute the eye aspect ratio
        ear = (A + B) / (2.0 * C)

        # Return the eye aspect ratio
        return ear
    

    def detect_blinks(self, cap, num_blinks_required: int = 2):
        try:
            # Initialize dlib's face detector and the facial landmark predictor
            detector = dlib.get_frontal_face_detector()
            predictor = dlib.shape_predictor("shape_predictor_68_face_landmarks.dat")

            # Initialize blink counter
            blink_counter = 0

            # Initialize variables for blink detection
            EYE_AR_THRESH = 0.27
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
                    shape = [(shape.part(i).x, shape.part(i).y) for i in range(68)]

                    left_eye = shape[42:48]
                    right_eye = shape[36:42]

                    left_ear = self.eye_aspect_ratio(left_eye)
                    right_ear = self.eye_aspect_ratio(right_eye)

                    ear = (left_ear + right_ear) / 2.0
                   
                    if ear < EYE_AR_THRESH:
                        COUNTER += 1
                    else:
                        if COUNTER >= EYE_AR_CONSEC_FRAMES:
                            TOTAL += 1
                        COUNTER = 0
                        
                if TOTAL >= num_blinks_required:
                    return True, ""
                
            cap.release()
            cv2.destroyAllWindows()

            return False, "Uh-oh! Blink and you'll miss it! Try blinking a bit more next time. 😉"
        except Exception as e:
            return False, str(e)



class FaceRecognition:

    def __init__(self, username: str, the_type: str, filename: str, video) -> None:
        self._bucketpath = os.getenv("BUCKETPATH", "face_recognition_v3/testing/encoding")
        self._video = video
        self._username = username
        self._type = the_type
        self._filename = filename
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
            f.write(self._video.read())

        return video_path

    def detect_faces(self, image):
        # Convert the image to grayscale
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

        # Detect faces in the image
        faces = self._face_cascade.detectMultiScale(gray, scaleFactor=1.3, minNeighbors=5)

        return faces

    def enroll(self) -> tuple:
        try:
            self.get_path()
            video_path = self.save_video()
            
            status, message = self.anti_spoof_liveliness(file_path=video_path)
            if not status:
                os.remove(video_path) if os.path.isfile(video_path) else None
                return True, message     

            cap = cv2.VideoCapture(video_path)
            count = 0

            output_dir = Path(self.IMAGEPATH + f"/{self._username}")
            output_dir.mkdir(exist_ok=True)

            while True:
                status, frame = cap.read()
                if not status:
                    break

                # Detect faces in the frame
                faces = self.detect_faces(frame)

                for (x, y, w, h) in faces:
                    # Save the face region as an image
                    
                    image_file = str(output_dir) + "/{count}.jpg".format(count=count + 1)

                    face_image = frame[y:y + h, x:x + w]
                    cv2.imwrite(image_file, face_image)
                    count += 1


            cap.release()
            cv2.destroyAllWindows()

            os.remove(video_path) if os.path.exists(video_path) else None

            self.save_faces_to_pickle(images_dir=str(output_dir))
            return False, "Enrollment Successful"
        except Exception as e:
            return True, str(e)


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

        # SEND FILE TO GCP in face_recognition
        try:
            storage_client = storage.Client()
            bucket = storage_client.bucket('face_recognition_v3')
            blobs = storage_client.list_blobs(f'{self._bucketpath}/encoding')
            blob = bucket.blob(f'{self._bucketpath}/encoding/{self._username}.pkl')
            with open(pickle_file_path, 'rb') as f:
                blob.upload_from_file(f)
        except Exception as e:
            print(str(e))

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
        

    def verify(self):
        try:
            self.get_path()
            video_path = self.save_video()
            
            status, message = self.anti_spoof_liveliness(file_path=video_path)
            if not status:
                os.remove(video_path) if os.path.isfile(video_path) else None
                return True, message            

            cap = cv2.VideoCapture(video_path)

            # Load enrolled faces from pickle file
            if not os.path.isfile(self.ENCODINGPATH + f"/{self._username}.pkl"):
                # Download pickle file if not available locally
                storage_client = storage.Client()
                bucket = storage_client.bucket('face_recognition_v3')
                blob = bucket.blob(f'{self._bucketpath}/encoding/{self._username}.pkl')
                blob.download_to_filename(self.ENCODINGPATH + f"/{self._username}.pkl")

            with open(self.ENCODINGPATH + f"/{self._username}.pkl", 'rb') as f:
                enrolled_faces = pickle.load(f)

            # Initialize LBPH face recognizer
            recognizer = cv2.face.LBPHFaceRecognizer_create()

            # Prepare enrolled faces and labels for training
            user_faces = enrolled_faces.get(self._username, [])
            label_mapping = {username: i for i, username in enumerate(enrolled_faces.keys())}
            labels_int = [label_mapping[self._username] for _ in range(len(user_faces))]
            gray_enrolled_faces = [cv2.cvtColor(face, cv2.COLOR_BGR2GRAY) for face in user_faces]

            # Train the LBPH model with enrolled faces and integer labels
            recognizer.train(gray_enrolled_faces, np.array(labels_int))

            recognized = 0
            unrecognized = 0

            while True:
                ret, frame = cap.read()
                if not ret:
                    break

                # Detect faces in the frame
                # Make sure to implement the detect_faces method properly
                faces = self.detect_faces(frame)

                for (x, y, w, h) in faces:
                    # Extract face region and convert to grayscale
                    face_image = cv2.cvtColor(frame[y:y + h, x:x + w], cv2.COLOR_BGR2GRAY)

                    # Perform face recognition using LBPH
                    label, confidence = recognizer.predict(face_image)


                    # Match against enrolled faces
                    if label != -1:  # Face recognized
                        if confidence < 50:
                            recognized += 1
                        else:
                            unrecognized += 1

                        # Break out of the loop after recognizing a face
                        break

            cap.release()
            cv2.destroyAllWindows()

            os.remove(video_path) if os.path.isfile(video_path) else None
            shutil.rmtree(f"{self.IMAGEPATH}/{self._username}", ignore_errors=True) if os.path.exists(
                f"{self.IMAGEPATH}/{self._username}") else None
            
            if unrecognized >= 50:
                return True, "Face Verification Failed"
            
            if recognized > unrecognized:
                return False, "Face verification Successful"

            return True, "Face Verification Failed"
            
        except Exception as e:
            return True, str(e)
        
    
    @staticmethod
    def anti_spoof_liveliness(file_path: str):
        try:
            anti_spoof = AntiSpoof(file_path=file_path)
            return anti_spoof.verify()
        except Exception as e:
            return True, str(e)
            