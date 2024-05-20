FROM python:3.10-slim

ENV PORT 5000
ENV APPDIR /app
ENV PYTHONUNBUFFERED True

WORKDIR $APPDIR

COPY . $APPDIR

RUN apt-get update && apt-get install -y cmake build-essential python3-opencv libpng-dev wget bzip2

# Download the shape_predictor_68_face_landmarks.dat.bz2 file
# RUN wget https://github.com/italojs/facial-landmarks-recognition/raw/master/shape_predictor_68_face_landmarks.dat

RUN pip install --upgrade pip

#  Install production dependencies
RUN pip install -r requirements.txt

# Single worker with 8 threads, timeout set at 30s
CMD exec gunicorn --bind 0.0.0.0:$PORT --workers 1 --threads 8 --timeout 300 app:app