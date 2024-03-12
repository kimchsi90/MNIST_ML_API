import tensorflow as tf
import numpy as np
import onnxruntime as rt
import mlflow
from typing import Union
from flask import Flask, request, jsonify
from flask_cors import CORS
from tensorflow.keras.callbacks import EarlyStopping
from mlflow.models.signature import ModelSignature
from mlflow.types.schema import Schema, TensorSpec
from utils.mlflow_utils import (
    get_latest_registered_model_uri,
    get_registered_model_info,
    parse_auto_logged_info,
)
from utils.model_utils import simple_classifier, convert_tf_to_onnx, validate_onnx_model


EXPERIMENT_NAME = "MNIST Simple CNN Experiment"
HOST = "0.0.0.0"
API_PORT = 5001
MODEL_NAME = "simple-cnn-mnist-reg-model"

# Set an experiment name
mlflow.set_experiment(EXPERIMENT_NAME)

# Instantiate the app
app = Flask(__name__)

# API의 역할을 하기 위해 다른 출처(Cross-Origin)에서 API에 접근 가능하도록 하는 역할
CORS(app)


def load_mnist_data() -> np.ndarray:
    # MNIST 데이터 로드 및 normalize
    f = np.load("mnist.npz")
    train_X, train_Y, test_X, test_Y = f['x_train'], f['y_train'], f['x_test'], f['y_test']
    f.close()
    train_X, test_X = train_X / 255.0, test_X / 255.0
    trainX = train_X.reshape((train_X.shape[0], 28, 28, 1))
    testX = test_X.reshape((test_X.shape[0], 28, 28, 1))
    trainY = tf.keras.utils.to_categorical(train_Y)  # ont-hot encoding
    testY = tf.keras.utils.to_categorical(test_Y)

    return trainX, testX, trainY, testY


# API train 엔드포인트 정의
@app.route("/train/", methods=["POST"])
def train() -> dict[str, Union[str, list[str], dict[str, Union[str, float]]]]:
    """
    입력받은 하이퍼파라미터(epochs, batch_size)를 기반으로 MNIST 분류를 위한 딥러닝 모델을 학습하고
    local experiment tracking server에 학습 metrics과 artifacts를 로깅하는 함수
    :return: tracked experiment entry에 대한 정보
    """
    data = request.get_json()
    epochs = int(data["epochs"])
    batch_size = int(data["batch_size"])
    trainX, testX, trainY, testY = load_mnist_data()
    with mlflow.start_run() as run:
        # autolog 시작
        mlflow.tensorflow.autolog()
        model = simple_classifier()
        # EarlyStopping 콜백 생성
        early_stopping = EarlyStopping(monitor="val_loss", patience=10)

        model.fit(
            trainX,
            trainY,
            epochs=epochs,
            batch_size=batch_size,
            validation_data=(testX, testY),
            callbacks=[early_stopping],
        )

        # autolog 종료
        mlflow.tensorflow.autolog(disable=True)

        info = parse_auto_logged_info(mlflow.get_run(run_id=run.info.run_id))
        
    return info


# API register 엔드포인트 정의
@app.route("/register/", methods=["POST"])
def register():
    """
    먼저 training experiment의 reference인 run id를 받아 해당하는 모델을 ONNX format으로 export하고 그 모델을 model registry에 promote한다.
    그리고 registered 모델에 대한 정보를 리턴하는 함수
    :return: registered model에 대한 정보
    """

    # run id를 이용하여 Mlflow Tracking Server에서 Tensorflow 모델을 로드
    data = request.get_json()
    run_id = data["run_id"]
    model_uri = f"runs:/{run_id}/model"
    loaded_tf_model = mlflow.tensorflow.load_model(model_uri)

    # 로드한 Tensorflow 모델을 ONNX 모델로 변환
    onnx_model = convert_tf_to_onnx(loaded_tf_model)
    # 변환 전/후의 결과를 비교하여 모델을 검증
    validate_onnx_model(loaded_tf_model, onnx_model)

    # MLflow에 ONNX 모델 로그, 기존 모델과 이름이 같을 경우 새로운 버전으로 등록된다.
    with mlflow.start_run():
        input_schema = Schema(
            [
                TensorSpec(np.dtype(np.float32), (-1, 28, 28, 1)),
            ]
        )
        output_schema = Schema([TensorSpec(np.dtype(np.float32), (-1, 10))])
        signature = ModelSignature(inputs=input_schema, outputs=output_schema)
        mlflow.onnx.log_model(
            onnx_model,
            artifact_path="model",
            registered_model_name=MODEL_NAME,
            signature=signature,
        )
        info = get_registered_model_info(MODEL_NAME)

    return info


# API predict 엔드포인트 정의
@app.route("/predict/", methods=["POST"])
def predict():
    """
    이미지 파일 1장을 받고 Mlflow Model Registry에 가장 최근에 등록된 모델을 이용하여 MNIST 이미지를 추론하고
    예측 결과를 리턴하는 함수
    :return: 예측 결과
    """
    data = request.get_json()
    image = np.array(data["image"]).reshape(1, 28, 28, 1)  # 이미지 reshape
    # 모델 이름을 이용하여 가장 최근에 등록된 model uri를 얻는 함수
    model_uri = get_latest_registered_model_uri(MODEL_NAME)  

    # model uri를 이용하여 MLflow Registry에 가장 최근에 등록된 ONNX 모델 로드
    onnx_model = mlflow.onnx.load_model(model_uri=model_uri)

    # ONNX model을 이용한 MNIST 이미지 결과 추론
    session = rt.InferenceSession(onnx_model.SerializeToString())
    input_name = session.get_inputs()[0].name
    pred = session.run(None, {input_name: image.astype(np.float32)})
    pred = np.argmax(pred)

    return jsonify({"predicted_digit": int(pred)})


@app.route('/health', methods=['GET'])
def health_check():

    return jsonify({"status": "ok"}), 200


# Run API server
if __name__ == "__main__":
    app.run(host=HOST, port=API_PORT, debug=False)
