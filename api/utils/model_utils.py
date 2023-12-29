import onnx
import onnxruntime as rt
import numpy as np
import tf2onnx
from tensorflow import TensorSpec, float32
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Dense, Flatten, Dropout
from tensorflow.keras.optimizers import Adam


def simple_classifier():
    model = Sequential()
    model.add(
        Conv2D(
            32,
            (3, 3),
            activation="relu",
            kernel_initializer="he_uniform",
            input_shape=(28, 28, 1),
        )
    )
    model.add(Conv2D(64, (3, 3), activation="relu"))
    model.add(MaxPooling2D((2, 2)))
    model.add(Dropout(0.25))
    model.add(Flatten())
    model.add(Dense(128, activation="relu", kernel_initializer="he_uniform"))
    model.add(Dropout(0.5))
    model.add(Dense(10, activation="softmax"))
    opt = Adam(learning_rate=0.01)
    model.compile(optimizer=opt, loss="categorical_crossentropy", metrics=["accuracy"])

    return model


def convert_tf_to_onnx(tf_model):
    spec = (
        TensorSpec((None, 28, 28, 1), float32, name="serving_default_conv2d_input"),
    )
    model_proto, _ = tf2onnx.convert.from_keras(tf_model, input_signature=spec)
    onnx.checker.check_model(model_proto)
    """모델의 구조를 확인하고 모델이 유효한 스키마(valid schema)를 가지고 있는지를 체크합니다. ONNX 그래프의 유효성은 모델의 버전,
    그래프 구조, 노드들, 그리고 입력값과 출력값들을 모두 체크하여 결정됩니다."""
    # 모델을 바이트로 변환
    onnx_model_bytes = model_proto.SerializeToString()

    return onnx_model_bytes


def validate_onnx_model(tf_model, onnx_model):
    # ONNX 모델 로드
    ort_session = rt.InferenceSession(onnx_model)

    input_name = ort_session.get_inputs()[0].name
    # 랜덤한 입력 데이터 생성
    input_data = np.random.normal(size=(1, 28, 28, 1)).astype(np.float32)

    # Tensorflow 모델 추론 결과값
    tf_out = tf_model.predict(input_data)

    # ONNX 런타임에서 계산된 추론 결과값
    ort_outs = ort_session.run(None, {input_name: input_data})

    # ONNX 런타임과 Tensorflow에서 연산된 결과값 근사적으로 동일한지 비교
    np.testing.assert_allclose(tf_out, ort_outs[0], rtol=1e-03, atol=1e-05)
