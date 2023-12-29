# syntax=docker/dockerfile:1
FROM python:3.10-slim

WORKDIR /app

COPY . .

RUN pip install -U pip &&\
    pip install -r requirements.txt &&\
    apt-get update && apt-get install -y curl