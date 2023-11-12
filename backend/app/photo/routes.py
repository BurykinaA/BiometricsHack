from app.photo import photo
from flask import request
import requests
from flask import current_app as app, make_response, jsonify, send_file
from flask_cors import cross_origin
import base64
import csv
import io
import os
import numpy as np
from PIL import Image
from MiniFasNet.test import get_sreenshot
from torchvision import transforms
import torch
from MCNN.inference import get_score
from MobileNet.inference import get_score_mobil
from Conv.inference import get_score_conv

# transform = transforms.Compose([
#         transforms.ToPILImage(),
#         transforms.Resize((128, 128)),
#         transforms.PILToTensor(),
#     ])

class SquareCrop:
    def __call__(self, image):
        return transforms.functional.center_crop(image, min(image.size))

transform = transforms.Compose([
    transforms.ToPILImage(),
    SquareCrop(), # Обрезка изображения до квадрата
    transforms.Resize((128, 128)), # Изменение размера изображения
    transforms.ToTensor(),
])


@cross_origin()
@photo.post("/api/photo")
def make_correction():
    data_list = request.json  # Теперь ожидаем список JSON объектов
    responses = []

    for data in data_list:
        image_bytes = base64.b64decode(data["photo"])
        image_io = io.BytesIO(image_bytes)
        image = Image.open(image_io)
        numpy_array = np.array(image)

        X = transform(numpy_array)
        X = X * 255
        X = X.unsqueeze(0)
        # print(X)

        print('---------')
        mcnn = 'real' if get_score(X) == 1 else 'fake'
        print(mcnn)
        mobilenet = 'real' if get_score_mobil(X) == 1 else 'fake'
        print(mobilenet)
        minifasnet = 'real' if get_sreenshot(numpy_array) == 1 else 'fake'
        print(minifasnet)
        # conv = 'real' if get_score_conv(numpy_array) == 1 else 'fake'
        # print(conv)

        resp = {'minifasnet': minifasnet, 'mcnn': mcnn, 'mobilenet':mobilenet, 'conv':'hui', "photo": data["photo"]}
        responses.append(resp)
        # print(responses)

    return make_response(responses)


@photo.post("/api/cam")
def camera_pic():
    data_list = request.json  # Теперь ожидаем список JSON объектов
    responses = []

    for data in data_list:
        image_bytes = base64.b64decode(data["photo"])
        image_io = io.BytesIO(image_bytes)
        image = Image.open(image_io)
        numpy_array = np.array(image)

        X = transform(numpy_array)
        X = X * 255
        X = X.unsqueeze(0)
        # print(X)

        mcnn = 'real' if get_score(X) == 1 else 'fake'
        print(mcnn)
        mobilenet = 'real' if get_score_mobil(X) == 1 else 'fake'
        print(mobilenet)
        minifasnet = 'real' if get_sreenshot(numpy_array) == 1 else 'fake'
        print(minifasnet)
        # conv = 'real' if get_score_conv(numpy_array) == 1 else 'fake'
        # print(conv)

        resp = {'minifasnet': minifasnet, 'mcnn': mcnn, 'mobilenet':mobilenet, 'conv':'hui', "photo": data["photo"]}
        responses.append(resp)
        # print(responses)

    return make_response(responses)
