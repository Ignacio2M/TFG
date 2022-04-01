import cv2 as cv
import os

# imageenes te pruebas
test_input_path = "Test_images"
# Imagenes preporcesadas
test_out_pre_path = "Test_images/OutPut/Pre"
# Resultados de la red
test_net_path = "Test_images/OutPut/Net"
# Resultados de SR
test_SR_path = "Test_images/Final"


def load_image(path, sub_dir, cv_read_module: cv.cv2 = None, function=None):
    img_dict = {}
    for name in sub_dir:
        img_dict[name] = []
        aux_path = os.path.join(path,name)
        list_path = os.listdir(aux_path)

        for image_name in list_path:
            if function is None:
                img_dict[name].append(cv.imread("{}/{}".format(aux_path, image_name), flags=cv_read_module))
            else:
                img_dict[name].append(function(cv.imread("{}/{}".format(aux_path, image_name), flags=cv_read_module)))

    return img_dict


def save(path, dir_images, images, names):
    path = os.path.join(path, dir_images)
    if isinstance(images, list) and isinstance(names, list):
        info_image = zip(images, names)
    else:
        info_image = zip([images], [names])
    if not os.path.exists(path):
        os.makedirs(path)
    for index, image_data in enumerate(info_image):
        (image, name) = image_data
        cv.imwrite("{}/{}.png".format(path, name), image)
