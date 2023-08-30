FROM python:3.10-slim

ENV PORT 8080
ENV APPDIR /app
ENV PYTHONUNBUFFERED True

WORKDIR $APPDIR

COPY . $APPDIR

RUN apt-get update && apt-get install -y cmake build-essential python3-opencv libpng-dev
RUN pip install --upgrade pip
#  Install production dependencies
RUN pip install -r requirements.txt

# Single worker with 8 threads, timeout set at 30s
CMD exec gunicorn --bind 0.0.0.0:$PORT --workers 1 --threads 8 --timeout 30 app:app