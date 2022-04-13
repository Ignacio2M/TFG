import datetime
import math
import re
import uuid

import cv2 as cv
import os
import numpy as np
from matplotlib import pyplot

import Utils.file_tools as ft
import json


class SR:
    init_load_path: str
    init_save_path: str
    final_load_path: str
    final_save_path: str
    info_metric_path: str
    rotation_angle: int
    backGroundImage: np.array

    def __init__(self,
                 uuid: str,
                 init_load_path: str,
                 init_save_path: str,
                 final_load_path: str,
                 final_save_path: str,
                 info_metric_path: str,
                 rotate_increment: int,
                 translate_vecto: np.array) -> None:

        self.final_save_path = os.path.join(final_save_path, uuid)
        self.final_load_path = os.path.join(final_load_path, uuid)
        self.init_save_path = os.path.join(init_save_path, uuid)
        self.info_metric_path = os.path.join(info_metric_path, uuid)
        self.init_load_path = init_load_path

        self.rotation_angle = rotate_increment
        self.translate_vecto = translate_vecto

        with open("Test_images/info.json", "r") as json_file:
            info = json.load(json_file)

        if uuid in info["SR_info"].keys():
            info_dict = info["SR_info"][uuid]
            info_dict["Date"] = datetime.datetime.now().strftime("%d/%m/%Y %H:%M:%S")
        else:
            info_dict = {
                "Date": datetime.datetime.now().strftime("%d/%m/%Y %H:%M:%S"),
                "Path": {
                    "Dir": [],
                    "Load_original": self.init_load_path,
                    "Save_pre": self.init_save_path,
                    "Save_net": self.final_load_path,
                    "Save_final": self.final_save_path,
                    "Save_metric": self.info_metric_path,
                },
                "Pre": {}
            }

        self.uuid = uuid
        self.info_dict = info_dict
        # self._Save_info()

    def _Save_info(self):
        with open("Test_images/info.json","r+") as json_file:
            info = json.load(json_file)
            info["SR_info"][self.uuid] = self.info_dict
            json_file.seek(0)  # rewind
            json.dump(info, json_file, indent=2)
            json_file.truncate()

    def init_sr(self, images_path: list, num_samples, **kwargs):

        # init_load_path
        #     └ images_dir
        #         ├ image.png
        #         └ image.png

        # _________________ kwargs __________________
        angle_space_const = kwargs.get("angle_space_const", True)
        angle_time_const = kwargs.get("angle_time_const", True)

        shift_space_const = kwargs.get("shift_space_const", False)
        shift_time_const = kwargs.get("shift_time_const", True)

        # _________________ Guardado de información _________________
        self.info_dict["Path"]["Dir"] = images_path
        self.info_dict["Pre"] = {
            "Num_samples": num_samples,
            "Constants": {
                "angle_time_const": angle_space_const,
                "angle_space_const": angle_time_const,

                "shift_space_const": shift_space_const,
                "shift_time_const": shift_time_const,
            },
            "Samples": {}
        }

        samples_info = self.info_dict["Pre"]["Samples"]

        # Nivel 1 de las carpetas
        for index_dir, images_dir in enumerate(images_path):
            aux_path = os.path.join(self.init_load_path, images_dir)
            list_img_path = os.listdir(aux_path)

            load_image_dir = "{}/{}".format(index_dir, len(images_dir))

            initial_shape = np.array(ft.image_load(path=aux_path, name=list_img_path[0])).shape[:2]
            final_shape = (self.translate_vecto * num_samples) + \
                          np.full((1, 2),
                                  math.sqrt(math.pow(initial_shape[0] / 2, 2) + math.pow(initial_shape[1] / 2, 2)) * 2,
                                  dtype=int)[0]
            final_shape = [final_shape.tolist()[1], final_shape.tolist()[0]]

            samples_info[images_dir] = {
                "Init_shape": initial_shape,
                "Final_shape": final_shape,
                "Samples": []
            }

            image_sample_ingo = samples_info[images_dir]["Samples"]

            angle = 0
            translate_vecto = np.array([0, 0])

            for sample in range(num_samples):
                # ______ Inicializo angulo y desplazamiento __________
                if not angle_space_const:
                    angle += self.rotation_angle
                else:
                    angle = self.rotation_angle
                aux_angle = angle

                if not shift_space_const:
                    translate_vecto += self.translate_vecto
                else:
                    translate_vecto = self.translate_vecto
                aux_translate_vecto = translate_vecto

                list_sample = []
                # _____ Imagenes _______
                for index_image, img_path in enumerate(list_img_path):
                    if not angle_time_const:
                        aux_angle += self.rotation_angle

                    if not shift_time_const:
                        aux_translate_vecto += self.translate_vecto

                    image = ft.image_load(path=aux_path, name=img_path)
                    point, image = self._rotate(image, angle, (translate_vecto * -1), final_shape)

                    # ____ Guadado de imagen _____
                    ft.save(
                        path=os.path.join(self.init_save_path + '/' + images_dir, "{}_{}".format(images_dir, sample)),
                        name="{}".format(index_image),
                        image=image
                    )
                    # ____ Informacon del sample _____
                    list_sample.append({
                        "translate_vector": aux_translate_vecto.tolist(),
                        "angle": aux_angle,
                        "points": point.tolist()
                    })

                # ___ Guardado de información
                image_sample_ingo.append(list_sample)
            # ____ Guardador de información
            samples_info[images_dir]["Samples"] = image_sample_ingo
        # ____ Guardado de información
        self.info_dict["Pre"]["Samples"] = samples_info

        # ___ Escrtura de la información ____
        self._Save_info()

    def _rotate(self, image, angle, translate_vector, final_shape, points=None):
        transformation_matrix = self._transform_matrix(angle, translate_vector)
        nrows, ncols, _ = image.shape
        original_corners = np.array([[0, 0, 1], [ncols, 0, 1], [ncols, nrows, 1], [0, nrows, 1]]).T
        new_corners = np.int64(np.dot(transformation_matrix, original_corners))
        x, y, new_width, new_height = cv.boundingRect(new_corners.T.reshape(1, 4, 2))
        transformation_matrix[:, 2] = np.array([-x, -y])
        image = cv.warpAffine(image, transformation_matrix, final_shape)
        if points is None:
            new_corners = np.int64(np.dot(transformation_matrix, original_corners))
            return new_corners, image
        else:
            new_points = np.int64(np.dot(transformation_matrix, points))
            print(new_points.T[2, :] - new_points.T[0, :])
            pyplot.imshow(image)
            pyplot.show()
            return image

    def _un_rotate(self, image, angle, translate_vector, final_shape, points):
        transformation_matrix = self._transform_matrix(angle, translate_vector)
        nrows, ncols, _ = image.shape
        original_corners = np.array(points)
        new_corners = np.int64(np.dot(transformation_matrix, original_corners))
        x, y, new_width, new_height = cv.boundingRect(new_corners.T.reshape(1, 4, 2))
        transformation_matrix[:, 2] = np.array([-x, -y])
        image = cv.warpAffine(image, transformation_matrix, final_shape)
        return image

    def _transform_matrix(self, rotate_angle, translate_vector: np.array) -> np.array:
        rotate_angle = math.radians(rotate_angle)
        matrix_transform = np.array([
            [math.cos(rotate_angle), -math.sin(rotate_angle), translate_vector[1]],
            [math.sin(rotate_angle), math.cos(rotate_angle), translate_vector[0]],
        ], dtype=np.float64)
        return matrix_transform

    def reshape(self, image, max_samples):
        image = np.append(image,
                          np.zeros((self.translate_vecto[1] * max_samples, image.shape[1], 3), dtype=image.dtype),
                          axis=0)
        image = np.append(image,
                          np.zeros((image.shape[0], self.translate_vecto[0] * max_samples, 3), dtype=image.dtype),
                          axis=1)
        return image

    def build_image(self, images_path: list or None = None, test=False):

        if images_path is None:
            images_path = self.info_dict["Path"]["Dir"]
        info = self.info_dict["Pre"]["Samples"]

        for index_path_samples, path_samples_aux in enumerate(images_path):
            path_samples = os.path.join(self.final_load_path, path_samples_aux)
            list_samples = list(filter(lambda name: not "log" in name, os.listdir(path_samples)))
            image_list = None
            aux_info_image_dir = info[path_samples_aux]
            final_shape = np.array(aux_info_image_dir["Init_shape"]) * 4
            final_shape = [final_shape[1], final_shape[0]]
            for index_sample, path_smple_images in enumerate(list_samples):
                path_smple_images = os.path.join(path_samples, path_smple_images)
                list_path_images = os.listdir(path_smple_images)

                if image_list is None:
                    image_list = [None] * len(list_path_images)
                for index_img, img_path in enumerate(list_path_images):
                    aux_info = aux_info_image_dir["Samples"][index_sample][index_img]
                    points_init = np.full((3, 4), 1 / 4)
                    points_init[:2, :] = aux_info["points"]
                    points_init = points_init * 4
                    angle = 360 - aux_info["angle"]
                    translate_vector = np.array(aux_info["translate_vector"]) * 4
                    translate_vector = [translate_vector[1], translate_vector[0]]
                    image = ft.image_load(path=path_smple_images, name=img_path)
                    image = self._un_rotate(image, angle, (0, 0), final_shape, points_init)
                    # pyplot.imshow(image)
                    # pyplot.show()
                    if image_list[index_img] is None:
                        image_list[index_img] = image / len(list_samples)
                    else:
                        image_list[index_img] = image_list[index_img] + (image / len(list_samples))

                if test:
                    for image in image_list:
                        pyplot.imshow(image)
                        pyplot.show()

            for index_img, image in enumerate(image_list):
                ft.save(
                    path=os.path.join(self.final_save_path, path_samples_aux),
                    name="{}".format(index_img),
                    image=image
                )
