import math
import re

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
    rotation_angle: int
    backGroundImage: np.array

    def __init__(self,
                 init_load_path: str,
                 init_save_path: str,
                 final_load_path: str,
                 final_save_path: str,
                 rotate_increment: int,
                 translate_vecto: np.array) -> None:

        self.max_samples = None
        self.rotation_angle = rotate_increment
        self.final_save_path = final_save_path
        self.final_load_path = final_load_path
        self.init_save_path = init_save_path
        self.init_load_path = init_load_path
        self.translate_vecto = translate_vecto

    def init_sr(self, images_path: list, num_samples, take_range=None, **kwargs):

        # init_load_path
        #     └ images_dir
        #         ├ image.png
        #         └ image.pnf

        info_dict = {}
        # translate_vecto = self.translate_vecto
        self.max_samples = num_samples
        # Note: Guardo la informacion para la reconstruccion
        angle_space_const = kwargs.get("angle_space_const", True)
        angle_time_const = kwargs.get("angle_time_const", True)

        shift_space_const = kwargs.get("shift_space_const", False)
        shift_time_const = kwargs.get("shift_time_const", True)

        info_dict["n_samples"] = num_samples
        info_dict["kwargs"] = kwargs
        info_dict["Images_data"] = []
        # Nivel 1 de las carpetas
        for index_dir, images_dir in enumerate(images_path):
            aux_path = os.path.join(self.init_load_path, images_dir)
            list_img_path = os.listdir(aux_path)
            if take_range is not None:
                list_img_path = list_img_path[take_range[0]:take_range[1]]

            load_image_dir = "{}/{}".format(index_dir, len(images_dir))
            initial_shape = np.array(ft.image_load(path=aux_path, name=list_img_path[0])).shape[:2]
            final_shape = (self.translate_vecto * num_samples) + \
                          np.full((1, 2), math.sqrt(math.pow(initial_shape[0]/2, 2) + math.pow(initial_shape[1]/2, 2))*2,
                                  dtype=int)[0]
            final_shape=[final_shape.tolist()[1],final_shape.tolist()[0]]
            uax_info_dict = {
                "save_path": os.path.join(self.init_save_path, images_dir),
                "initial_shape": [initial_shape[1],initial_shape[0]],
                "final_shape": final_shape,
                "samples": []
            }
            angle = 0
            translate_vecto = np.array([0, 0])
            for sample in range(num_samples):
                load_samples = "{}/{}".format(sample, num_samples)
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
                for index_image, img_path in enumerate(list_img_path):
                    load_image = "{}/{}".format(index_image, len(list_img_path))
                    if not angle_time_const:
                        aux_angle += self.rotation_angle

                    if not shift_time_const:
                        aux_translate_vecto += self.translate_vecto

                    image = ft.image_load(path=aux_path, name=img_path)
                    point, image = self._rotate(image, angle, (translate_vecto * -1), final_shape)

                    ft.save(
                        path=os.path.join(self.init_save_path + '/' + images_dir, "{}_{}".format(images_dir, sample)),
                        name="{}".format(index_image),
                        image=image
                    )
                    list_sample.append({
                        "angle": aux_angle,
                        "translate_vector": aux_translate_vecto.tolist(),
                        "pointsImages": point.tolist()
                    })
                    # print("Images_dir: {}\nSample: {}\nImages: {}".format(load_image_dir, load_samples, load_image))
                aux = uax_info_dict["samples"]
                aux.append(list_sample)
                uax_info_dict["samples"] = aux
            aux = info_dict["Images_data"]
            aux.append(uax_info_dict)
            info_dict["Images_data"] = aux

        with open("{}/info.json".format(self.init_save_path), "w") as f:
            f.write(json.dumps(info_dict, indent=4))

    def _rotate(self, image, angle, translate_vector, final_shape, points=None):
        transformation_matrix = self._transform_matrix(angle, translate_vector)
        nrows, ncols, _ = image.shape
        original_corners = np.array([[0, 0, 1], [ncols, 0, 1], [ncols, nrows, 1], [0, nrows, 1]]).T
        new_corners = np.int64(np.dot(transformation_matrix, original_corners))
        x, y, new_width, new_height = cv.boundingRect(new_corners.T.reshape(1, 4, 2))
        # print(new_width, new_height)
        # print(transformation_matrix)
        transformation_matrix[:, 2] = np.array([-x, -y])
        # print(transformation_matrix)
        image = cv.warpAffine(image, transformation_matrix, final_shape)

        # pyplot.imshow(image)
        # pyplot.show()

        if points is None:
            new_corners = np.int64(np.dot(transformation_matrix, original_corners))
            return new_corners, image
        else:
            new_points = np.int64(np.dot(transformation_matrix, points))
            print(new_points.T[2,:]-new_points.T[0,:])
            # image = image[]
            pyplot.imshow(image)
            pyplot.show()
            return image

    def _un_rotate(self, image, angle, translate_vector, final_shape, points):
        transformation_matrix = self._transform_matrix(angle, translate_vector)
        nrows, ncols, _ = image.shape
        original_corners = np.array(points)
        new_corners = np.int64(np.dot(transformation_matrix, original_corners))
        x, y, new_width, new_height = cv.boundingRect(new_corners.T.reshape(1, 4, 2))
        # print(new_width, new_height)
        # print(transformation_matrix)
        transformation_matrix[:, 2] = np.array([-x, -y])
        # print(transformation_matrix)
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

    def build_image(self, images_path: list, test=False):

        with open('{}/info.json'.format(self.init_save_path)) as json_file:
            info = json.load(json_file)

        for index_path_samples, path_samples in enumerate(images_path):
            path_samples = os.path.join(self.final_load_path, path_samples)
            list_samples = os.listdir(path_samples)
            image_list = None
            for index_sample, path_smple_images in enumerate(list_samples):
                path_smple_images = os.path.join(path_samples, path_smple_images)
                list_path_images = os.listdir(path_smple_images)

                if image_list is None:
                    image_list = [None]*len(list_path_images)

                for index_img, img_path in enumerate(list_path_images):
                    aux_info = info["Images_data"][index_path_samples]["samples"][index_sample][index_img]
                    points_init = np.full((3, 4), 1 / 4)
                    points_init[:2, :] = aux_info["pointsImages"]
                    points_init = points_init * 4
                    angle = 360 - aux_info["angle"]
                    translate_vector = np.array(aux_info["translate_vector"])*4
                    translate_vector = [translate_vector[1], translate_vector[0]]
                    final_shape = np.array(info["Images_data"][index_path_samples]["initial_shape"])*4
                    image = ft.image_load(path=path_smple_images, name=img_path)
                    image = self._un_rotate(image, angle, (0,0), final_shape, points_init)
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
                    path=os.path.join(self.final_save_path, path_samples),
                    name="{}".format(index_img),
                    image=image
                )

                    # list_items_path = os.listdir()
        # with open('{}/info.json'.format(self.init_save_path)) as json_file:
        #     info = json.load(json_file)
        # image_list = []
        #
        # for index_img_dir, dir_path in enumerate(list_items_path):
        #     aux_path = os.path.join(path, dir_path)
        #     images_path = os.listdir(aux_path)
        #     # note: Lectura de las imagenes
        #     for index_image, img_path in enumerate(images_path):
        #         aux_info = info["Images_data"][index_img_dir]["samples"][index_image]
        #         image = ft.image_load(path=aux_path, name=img_path)
        #         image = self.unshift(image, index_image)
        #
        #         if image_list[index_image] is None:
        #             image_list[index_image] = image / self.max_samples
        #         else:
        #             image_list[index_image] = image_list[index_image] + (image / self.max_samples)

# class OP:
#
#     def __init__(self, shift_vector, max_num_samples=None):
#         self.Shift_vector = shift_vector
#         self.Num_samples = None
#         self.scale_factor = 4
#         self.shape = None
#         self.index_sample = None
#
#     def reshape(self, image):
#         image = np.append(image,
#                           np.zeros((self.Shift_vector[1] * self.Num_samples, image.shape[1], 3), dtype=image.dtype),
#                           axis=0)
#         image = np.append(image,
#                           np.zeros((image.shape[0], self.Shift_vector[0] * self.Num_samples, 3), dtype=image.dtype),
#                           axis=1)
#         return image
#
#     def shift(self, image):
#         if self.shape is None:
#             self.shape = image.shape
#         image = np.roll(np.roll(image, self.Shift_vector[1] * self.index_sample, axis=0),
#                         self.Shift_vector[0] * self.index_sample, axis=1)
#         return image
#
#     def un_shift(self, image, index=None, dtype=np.float32):
#         if index is None:
#             index = self.index_sample
#
#         init_point = (self.Shift_vector[0] * self.scale_factor * index,
#                       self.Shift_vector[1] * self.scale_factor * index)
#         final_point = (init_point[0] + (self.shape[0] * self.scale_factor),
#                        init_point[1] + (self.shape[1] * self.scale_factor))
#
#         return np.array(image[init_point[1]:final_point[1], init_point[0]:final_point[0], ::], dtype=dtype)
#
#
# class SR:
#     save_path = None
#     images_phat: list
#     shift_vecto = None
#     num_samples = None
#     path: str
#     operators: OP
#
#     def __init__(self, shift_vector):
#         self.shift_vecto = shift_vector
#         self.operators = OP(self.shift_vecto)
#         # self.path_dict = {}
#         # self.save_path = ft.test_out_pre_path
#
#     def shift(self, path, save_path, names, n_samples=5, is_video=False):
#         self.operators.Num_samples = n_samples
#         self.num_samples = n_samples
#         if save_path is not None:
#             self.save_path = save_path
#
#         if not isinstance(names, list):
#             names = [names]
#
#         image_dict = {}
#
#         for name in names:
#             alfa_path = os.path.join(path, name)
#             data_name_list = os.listdir(alfa_path)
#
#             name = name.split(".")[0]
#
#             if is_video:
#                 image_dict[name] = ft.load_video(alfa_path, data_name_list[0], function=self.operators.reshape)
#             else:
#                 image_dict[name] = (ft.load_image(alfa_path, data_name_list, function=self.operators.reshape))
#
#         for i in range(self.num_samples):
#             self.operators.index_sample = i
#             ft.save(
#                 path=save_path,
#                 data_dict=image_dict,
#                 function=self.operators.shift,
#                 aux_num_iter=i
#             )
#
#     # def shit(self, save_path, n_samples=5, shape_input=None, is_video=False):
#     #
#     #     self.operators = OP(self.shift_vecto, n_samples, shape_input)
#     #
#     #     if is_video:
#     #         image_dict = ft.load_video(self.path, self.images_phat, function=self.operators.reshape)
#     #     else:
#     #         image_dict = ft.load_image(self.path, self.images_phat, function=self.operators.reshape)
#     #
#     #     for name, image_list in image_dict.items():
#     #         aux_save_path = save_path
#     #         for index_sample in range(n_samples):
#     #             for index, image in enumerate(image_list):
#     #                 self.list_save_path.append("{}_{}".format(name, index_sample))
#     #                 ft.save(path=aux_save_path,
#     #                         dir_images="{}_{}".format(name, index_sample),
#     #                         images=self.operators.shift(image, index_sample),
#     #                         names="{}_{}".format(name, index))
#
#     def build_image(self, load_path, save_path, names_path: list):
#         image_path_dict = {}
#
#         for sub_path in names_path:
#             alfa_path = os.path.join(load_path, sub_path)
#             image_path_list = os.listdir(alfa_path)
#             image = None
#             for index, image_path in enumerate(image_path_list):
#                 aux_path = os.path.join(alfa_path, image_path)
#                 image_name_list = os.listdir(aux_path)
#                 print(index)
#                 self.operators.index_sample = index
#                 if image is None:
#                     image = np.stack(ft.load_image(aux_path, image_name_list, function=self.operators.un_shift)) \
#                             / self.operators.Num_samples
#                 else:
#                     image = image + (np.stack(ft.load_image(aux_path, image_name_list, function=self.operators.un_shift))\
#                             / self.operators.Num_samples)
#
#
#             image_path_dict[sub_path] = image
#
#         ft.save(path=save_path,
#                 data_dict=image_path_dict
#         )
#
#
