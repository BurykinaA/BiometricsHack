import os
import cv2
import numpy as np
import warnings
import time

from .anti_spoof_predict import AntiSpoofPredict
from .generate_patches import CropImage
from .utility import parse_model_name

warnings.filterwarnings("ignore")


def check_image(image):
    height, width, channel = image.shape
    if width / height != 3 / 4:
        return False
    else:
        return True


def test(numpy_array, model_dir, device_id):
    model_test = AntiSpoofPredict(device_id)
    image_cropper = CropImage()
    image = cv2.cvtColor(numpy_array, cv2.COLOR_RGB2BGR)
    result = check_image(image)
    if result is False:
        height, width, channel = image.shape
        new_width = int(height * (3 / 4))
        aspect_ratio = width / height

        new_width = int(height * aspect_ratio)
        new_height = height

        image = cv2.resize(image, (new_width, new_height))
    image_bbox = model_test.get_bbox(image)
    prediction = np.zeros((1, 3))
    test_speed = 0
    for model_name in os.listdir(model_dir):
        h_input, w_input, model_type, scale = parse_model_name(model_name)
        param = {
            "org_img": image,
            "bbox": image_bbox,
            "scale": scale,
            "out_w": w_input,
            "out_h": h_input,
            "crop": False,
        }
        if scale is None:
            param["crop"] = False
        img = image_cropper.crop(**param)
        start = time.time()
        prediction += model_test.predict(img, os.path.join(model_dir, model_name))
        test_speed += time.time() - start
        # print(test_speed)

    # draw result of prediction
    label = np.argmax(prediction)
    # print(label)
    return label


def get_sreenshot(img):
    """
    1 - real
    0 - fake
    """
    current_dir = os.path.dirname(os.path.abspath(__file__))
    relevant_path = "anti_spoof_models"
    path = os.path.join(current_dir, relevant_path)
    return test(
        img,
        path,
        0,
    )
