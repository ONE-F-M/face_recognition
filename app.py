import json
from flask import Flask, request, jsonify
from face_engine import Detector, set_credential

app = Flask(__name__)

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

    
if __name__ == "__main__":
    app.run(debug=True)