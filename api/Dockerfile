# syntax=docker/dockerfile:1
FROM python:3.10-slim

WORKDIR /api

COPY . .

RUN pip install -U pip &&\
    pip install -r requirements.txt &&\
    pip install tensorflow==2.11 &&\
    pip install protobuf==3.20.2 &&\
    apt-get update && apt-get upgrade -y && apt-get install -y git &&\
    apt-get install -y curl

CMD ["python", "api.py"]