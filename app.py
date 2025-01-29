import json, os, requests, bz2
from traceback import format_exc

from dotenv import load_dotenv
from flasgger import Swagger
from flask import Flask, request, jsonify
from face_engine import Detector, set_credential, FaceRecognition
from flask_cors import CORS



load_dotenv()

app = Flask(__name__)
swagger = Swagger(app)
CORS(app, origins=os.getenv('WHITELISTED_URLS', "").split(','))

save_path = os.path.join(os.getcwd(), 'shape_predictor_68_face_landmarks.dat')
if not os.path.isfile(save_path):
  response = requests.get('https://github.com/italojs/facial-landmarks-recognition/raw/master/shape_predictor_68_face_landmarks.dat', stream=True)

  compressed_file_path = os.path.join(os.getcwd(), 'shape_predictor_68_face_landmarks.dat')

  # Check if the request was successful
  if response.status_code == 200:
      with open(compressed_file_path, 'wb') as f:
          for chunk in response.iter_content(chunk_size=8192):
              f.write(chunk)
else:
    print("File exist")

if all((not os.path.isfile("cred.json"), os.getenv("GOOGLE_CREDENTIALS", ""))):
    with open("cred.json", "w") as new_file:
      new_file.write(os.getenv("GOOGLE_CREDENTIALS"))


@app.route("/")
def home():
    """Home Endpoint.
    ---
    responses:
      200:
        description: "Hello, World!"

    """

    return "Hello, World!"

@app.route('/bigbang', methods=['POST'])
def bigbang():
    data = request.get_json()
    if not (data.get('cred') and data.get('bucketpath')):
        return jsonify({'error':True, 'message':'Blackhole, Dead Star.'})
    if not (data['cred'].get('private_key_id') and data['cred'].get('project_id')):
        return jsonify({'error':True, 'message':'Blackhole, Dead Star.'})
    return jsonify(set_credential(data))


@app.route("/enroll", methods=['POST'])
def enroll():
    """Enrollment Endpoint
    This is used to enroll the user (Request should be sent as form-data).
    ---
    parameters:
      - name: username
        type: string
        required: true

      - name: filename
        type: string
        required: true

      - name: video_file
        type: file
        required: true

    definitions:
      enroll:
        type: object
        properties:
          error:
            type: boolean
          message:
            type: string

    responses:
      200:
        schema:
          $ref: '#/definitions/enroll'
    """
    data = request.form.to_dict()
    video = request.files.get("video_file")
    face_recogniton = FaceRecognition(username=data["username"], the_type="enroll", filename=data["filename"], video=video, decrypt_video = data['decrypt_video'])
    error, message , traceback = face_recogniton.enroll()
    return dict(error=error, message=message, traceback=traceback)



@app.route("/verify", methods=['POST'])
def verify():
    """Verification Endpoint
    This is used to verify the user (Request should be sent as form-data).
    ---
    parameters:
      - name: username
        type: string
        required: true

      - name: filename
        type: string
        required: true

      - name: video_file
        type: file
        required: true

    definitions:
      verification:
        type: object
        properties:
          error:
            type: boolean
          message:
            type: string

    responses:
      200:
        schema:
          $ref: '#/definitions/verification'

    """
    data = request.form.to_dict()
    video = request.files.get("video_file")
    face_recogniton = FaceRecognition(username=data["username"], the_type="verify", filename=data["filename"], video=video, decrypt_video = data['decrypt_video'])
    error, message, traceback = face_recogniton.verify()
    return dict(error=error, message=message, traceback=traceback)

@app.route('/shape-model-download', methods=['GET'])
def download_file():
    try:
        save_path = os.path.join(os.getcwd(), 'shape_predictor_68_face_landmarks.dat')
        if not os.path.isfile(save_path):
          response = requests.get('https://github.com/italojs/facial-landmarks-recognition/raw/master/shape_predictor_68_face_landmarks.dat', stream=True)

          compressed_file_path = os.path.join(os.getcwd(), 'shape_predictor_68_face_landmarks.dat')


          # Check if the request was successful
          if response.status_code == 200:
              with open(compressed_file_path, 'wb') as f:
                  for chunk in response.iter_content(chunk_size=8192):
                      f.write(chunk)

              return dict(error=False, message="File Downloaded Successfully", traceback="")
          return dict(error=True, message="Error while getting the file", traceback="")
        return dict(error=False, message="File Already Exist", traceback="")

    except Exception as e:
        return dict(error=True, message=str(e), traceback=str(format_exc()))

if __name__ == "__main__":
    app.run(debug=os.getenv('DEBUG', True), port=5000, host='0.0.0.0')