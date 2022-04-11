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
test_out_pre_path = "Test_images/OutPut/Pre"
# Resultados de la red
test_net_path = "Test_images/OutPut/Net"
# Resultados de SR
test_SR_path = "Test_images/Final"


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

    # def load_image(path, name: list, cv_read_module: cv.cv2 = None, function=None):
    #     image_list = []
    #     for aux_name in name:
    #         aux_path = os.path.join(path, aux_name)
    #         if os.path.isdir(aux_path):
    #             list_path = os.listdir(aux_path)
    #         else:
    #             aux_path = path
    #             list_path = [aux_name]
    #
    #         for image_name in list_path:
    #             if function is None:
    # image_list.append(cv.imread("{}/{}".format(aux_path, image_name), flags=cv_read_module))


#             else:
#                 image_list.append(function(cv.imread("{}/{}".format(aux_path, image_name), flags=cv_read_module)))
#
#     return image_list
#
#
# def load_video(path, name, function=None, range_frame=None):
#
#     aux_path = os.path.join(path, name)
#     cap = cv.VideoCapture(aux_path)
#
#     num_frame = cap.get(cv.CAP_PROP_FRAME_COUNT)
#
#     frame_lst = []
#
#     if range_frame is None or len(range_frame) != 2:
#         range_frame = (0, num_frame)
#     init, final = range_frame
#
#     for i in range(int(num_frame)):
#         ret, frame = cap.read()
#         if init <= i <= final:
#             if function is not None:
#                 frame = function(frame)
#             frame_lst.append(frame)
#
#     return frame_lst

def save(path, name, image):
    if not os.path.exists(path):
        os.makedirs(path)

    cv.imwrite("{}/{}.png".format(path, name), image)


# def save(path, data_dict: dict, function=None, aux_num_iter=None):
#     for name, data in data_dict.items():
#         if aux_num_iter is not None:
#             name = "{}/{}".format(name, aux_num_iter)
#         aux_path = os.path.join(path, name)
#         if not os.path.exists(aux_path):
#             os.makedirs(aux_path)
#         for index, image in enumerate(data):
#             if function is not None:
#                 image = function(image)
#             cv.imwrite("{}/{}_{}.png".format(aux_path, "out", index), image)
