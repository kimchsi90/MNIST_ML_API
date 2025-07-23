# MNIST Model Serving APIs

### e-mail: <kimchsi9004@naver.com>, mobile: 010-4058-8210

## 기술 요소

Docker

- Docker-compose (Container management)

Python

- Flask (API)
- Tensorflow, ONNX, ONNXRuntime (Machine learning)

Deep Learning
 - MLflow

# API 명세서

## 1. POST /train/{epochs, batch_size}

### Description

1. 모델 학습에 필요한 하이퍼파라미터(epoch 수, batch size)를 받아 MNIST 분류를 위한 딥러닝 모델을 학습한다.
2. MLflow 로컬 experiment tracking server에 metrics와 artifacts를 로깅한다.
3. 학습이 완료되면 추적된 학습 정보를 반환한다.

### Parameters

- `epochs` (required): train set을 반복 학습할 수
- `batch_size` (required): 한 번에 학습할 데이터의 수

### Example Request

{'epochs': 5, 'batch_size': 128}

## 2. POST /register/{run_id}

### Description

1. 학습 실험의 reference인 run id를 받아 해당하는 모델을 ONNX format으로 export한다.
2. 변환된 ONNX 모델을 검증한다.(Tensorflow 모델과의 prediction 결과 값의 차이를 확인): Model governence process
3. ONNX 모델을 Mlflow Model Registry에 등록한다.
4. 등록된 모델의 정보를 반환한다.

### Parameters

- `run_id` (required): 머신러닝 실험의 id

### Example Request

{'run_id': fd10a17d028c47399a55ab8741721ef7}

## 3. POST /predict/{image}

### Description

1. 이미지 파일을 받고, 가장 최근에 모델 레지스트리에 등록된 모델을 이용하여 숫자를 추론한다.
2. 예측 결과를 반환한다.

### Parameters

- `image` (required): 28x28x1 이미지

### Example Request

{'image': [[...], ..., [...]]}

## Docker/Docker-compose 사용하여 실행하기

* API에 요청하는 샘플 코드: api_request.py(train, register, predict API 호출)

- 애플리케이션 실행하기: `docker compose up`

- MLflow ui 접속: `https://localhost:5000/`

- 애플리케이션 멈추기: `docker compose stop`

- 애플리케이션 다시 띄우기: `docker compose start`

- 애플리케이션 삭제: `docker compose down`
