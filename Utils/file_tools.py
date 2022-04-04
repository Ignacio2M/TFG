import cv2 as cv
import os


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


def load_image(path, name: list, cv_read_module: cv.cv2 = None, function=None):
    image_list = []
    for aux_name in name:
        aux_path = os.path.join(path, aux_name)
        if os.path.isdir(aux_path):
            list_path = os.listdir(aux_path)
        else:
            aux_path = path
            list_path = [aux_name]

        for image_name in list_path:
            if function is None:
                image_list.append(cv.imread("{}/{}".format(aux_path, image_name), flags=cv_read_module))
            else:
                image_list.append(function(cv.imread("{}/{}".format(aux_path, image_name), flags=cv_read_module)))

    return image_list


def load_video(path, name, function=None, range_frame=None):

    aux_path = os.path.join(path, name)
    cap = cv.VideoCapture(aux_path)

    num_frame = cap.get(cv.CAP_PROP_FRAME_COUNT)

    frame_lst = []

    if range_frame is None or len(range_frame) != 2:
        range_frame = (0, num_frame)
    init, final = range_frame

    for i in range(int(num_frame)):
        ret, frame = cap.read()
        if init <= i <= final:
            if function is not None:
                frame = function(frame)
            frame_lst.append(frame)

    return frame_lst


def save(path, data_dict: dict, function=None, aux_num_iter=None):
    for name, data in data_dict.items():
        if aux_num_iter is not None:
            name = "{}/{}".format(name, aux_num_iter)
        aux_path = os.path.join(path, name)
        if not os.path.exists(aux_path):
            os.makedirs(aux_path)
        for index, image in enumerate(data):
            if function is not None:
                image = function(image)
            cv.imwrite("{}/{}_{}.png".format(aux_path, "out", index), image)