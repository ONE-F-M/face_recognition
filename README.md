## Face Recognition for the world
This is a simple face recognition system written in python, it takes video, creates snapshots, generate pickle and store in GCP.
To verify, take video pass to the verify api and it will return success or failed response.

It is built on Flask Web Framework with endpoinf for enroll, verify and setup GCP credentials

## Run
1. Activate virtual Env and pip install requirements.
2. gunicorn -c dev.py #startup server
