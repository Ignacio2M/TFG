import cv2 as cv
import os

import numpy as np

video_format = [".avi"]
image_format = [".png"]


# Original
test_original_path = "Test_images/Original"
# imageenes te pruebas
test_input_path = "Test_images"
# Imagenes preporcesadas
test_out_pre_path = "Test_images/Output/Pre"
# Resultados de la red
test_net_path = "Test_images/Output/Net"
# Resultados de SR
test_SR_path = "Test_images/Final"
# Mediciones
test_metric_path = "Test_images/Final/Medidas"

def image_load(path, name) -> np.array:
    image = cv.imread(os.path.join(path, name))
    return image


def video_load(path, function=None, range_frame=None):
    cap = cv.VideoCapture(path)
    totalFrames = cap.get(cv.CAP_PROP_FRAME_COUNT)
    if range_frame is None:
        range_frame = (0, totalFrames)

    image_list = []
    for frame in range(range_frame[0], range_frame[1]):
        cap.set(1, frame)
        _, image = cap.read()
        if function is not None:
            image = function(image)
        image_list.append(image)

    return image_list


def _aux_load(path, **params) -> list or dict:
    image_list = []
    image_dict = {}  # En caso de ser un video.
    function = params.get("function", None)
    name_data_list = os.listdir(path)
    for name in name_data_list:
        aux_path = os.path.join(path, name)
        data_format = os.path.splitext(name)[-1]
        if data_format in image_format:
            image_list.append(image_load(aux_path, function=function))
        elif data_format in video_format:
            range_frame = params.get("range_frame", None)
            name = os.path.splitext(name)[0]
            image_dict["/{}".format(name)] = video_load(aux_path, function=function, range_frame=range_frame)
    return []


def load_data(load_path, load_sub_path: list = None, **params) -> dict:
    data_dict = {}
    if load_sub_path is not None:
        for sub_path in load_sub_path:
            path = os.path.join(load_path, sub_path)
            data_dict["/{}".format(sub_path)] = _aux_load(path, **params)
    else:
        name = os.path.split(load_path)[-1]
        data_dict[name] = _aux_load(load_path, **params)


def save(path, name, image):
    if not os.path.exists(path):
        os.makedirs(path)

    cv.imwrite("{}/{}.png".format(path, name), image)
