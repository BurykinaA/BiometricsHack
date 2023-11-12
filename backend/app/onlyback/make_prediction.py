import csv
import io
import os
import numpy as np
from PIL import Image

# from MiniFasNet.test import get_sreenshot
from torchvision import transforms
import torch

# from MCNN.inference import get_score
# from MobileNet.inference import get_score_mobil
# from Conv.inference import get_score_conv
from torchvision.io import read_image

transform = transforms.Compose(
    [
        transforms.ToPILImage(),
        transforms.Resize((128, 128)),
        transforms.ToPILImage(),
    ]
)


def make_correction():
    numpy_array = read_image(r"C:\Users\alina\Desktop\photo_2023-11-12_03-49-45.jpg")
    print(numpy_array)
    X = transform(numpy_array)
    X = torch.tensor(X).unsqueeze(0).float()
    # mcnn = 'real' if get_score(X) == 1 else 'fake'
    # print(mcnn)
    # mobilenet = 'real' if get_score_mobil(X) == 1 else 'fake'
    # print(mobilenet)
    # minifasnet = 'real' if get_sreenshot(numpy_array) == 1 else 'fake'
    # print(minifasnet)


if __name__ == "__main__":
    make_correction()
