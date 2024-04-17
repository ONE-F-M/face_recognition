import json, os

from dotenv import load_dotenv
from flask import Flask, request, jsonify, abort
from face_engine import Detector, set_credential, AntiSpoof
from flask_cors import CORS


load_dotenv()

app = Flask(__name__)
CORS(app, origins=os.getenv('WHITELISTED_URLS', "").split(',')) 

@app.route("/")
def home():
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
    data = request.get_json()
    # use detector
    detector = Detector(username=data['username'], bucketpath=data['bucketpath'])
    res = detector.enroll(video=data['video'], filename=data['filename'])
    return jsonify(res)

@app.route("/verify", methods=['POST'])
def verify():
    data = request.get_json()
    # print(data)
    # use detector
    detector = Detector(username=data['username'], bucketpath=data['bucketpath'])
    res = detector.verify(video=data['video'], filename=data['filename'])
    return jsonify(res)


@app.route("/anti-spoof", methods=["POST"])
def verify_spoof():
    file = request.files.get("video_file")
    if not file:
        abort(400, 'Missing Video File')
    antispoof = AntiSpoof(video_file=file)
    res = antispoof.verify()
    return jsonify(res)

    
if __name__ == "__main__":
    app.run(debug=True)