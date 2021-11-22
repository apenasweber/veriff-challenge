FROM python:3.8
WORKDIR /birdclassifier-app
COPY requirements.txt .
RUN apt-get update \
    && python -m pip install --upgrade pip \
    && pip install -r requirements.txt \
    && apt-get install ffmpeg libsm6 libxext6  -y
COPY . .
CMD ["python", "classifier.py"]