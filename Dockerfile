FROM python:3.10-slim

ENV PORT 5000
ENV APPDIR /app
ENV PYTHONUNBUFFERED True

WORKDIR $APPDIR

COPY . $APPDIR

RUN apt-get update && apt-get install -y cmake build-essential libpng-dev
RUN pip install --upgrade pip
RUN pip install -r requirements.txt

EXPOSE 5000
CMD ["python", "app.py", "--host", "0.0.0.0", "--port", "5000"]