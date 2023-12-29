import requests
import cv2

#command = "train"
#command = "register"
command = "predict"

epochs = 1
batch_size = 128
run_id = "ab3b1bd3583e4506b8d91eddfe54977b"
img_data = cv2.imread("sample_data/0.jpg", cv2.IMREAD_GRAYSCALE)


def train():
    url = "http://localhost:5001/train/"
    data = {'epochs': epochs, 'batch_size': batch_size}
    response = requests.post(url, json=data)
    print(response)
    print(response.json())


def register():
    url = "http://localhost:5001/register/"
    data = {'run_id': run_id}
    response = requests.post(url, json=data)
    print(response)
    print(response.json())


def predict():
    url = "http://localhost:5001/predict/"
    data = {'image': img_data.tolist()}
    response = requests.post(url, json=data)
    print(response)
    print(response.json())


def api_request(command):
    if command == "train":
        train()
    elif command == "register":
        register()
    elif command == "predict":
        predict()
    else:
        print("Unknown command!")


api_request(command)
