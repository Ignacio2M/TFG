import re

import cv2 as cv
import os
import numpy as np

import Utils.file_tools as ft


class OP:

    def __init__(self, shift_vector, max_num_samples=None):
        self.Shift_vector = shift_vector
        self.Num_samples = None
        self.scale_factor = 4
        self.shape = None
        self.index_sample = None

    def reshape(self, image):
        image = np.append(image,
                          np.zeros((self.Shift_vector[1] * self.Num_samples, image.shape[1], 3), dtype=image.dtype),
                          axis=0)
        image = np.append(image,
                          np.zeros((image.shape[0], self.Shift_vector[0] * self.Num_samples, 3), dtype=image.dtype),
                          axis=1)
        return image

    def shift(self, image):
        if self.shape is None:
            self.shape = image.shape
        image = np.roll(np.roll(image, self.Shift_vector[1] * self.index_sample, axis=0),
                        self.Shift_vector[0] * self.index_sample, axis=1)
        return image

    def un_shift(self, image, index=None, dtype=np.float32):
        if index is None:
            index = self.index_sample

        init_point = (self.Shift_vector[0] * self.scale_factor * index,
                      self.Shift_vector[1] * self.scale_factor * index)
        final_point = (init_point[0] + (self.shape[0] * self.scale_factor),
                       init_point[1] + (self.shape[1] * self.scale_factor))

        return np.array(image[init_point[1]:final_point[1], init_point[0]:final_point[0], ::], dtype=dtype)


class SR:
    save_path = None
    images_phat: list
    shift_vecto = None
    num_samples = None
    path: str
    operators: OP

    def __init__(self, shift_vector):
        self.shift_vecto = shift_vector
        self.operators = OP(self.shift_vecto)
        # self.path_dict = {}
        # self.save_path = ft.test_out_pre_path

    def shift(self, path, save_path, names, n_samples=5, is_video=False):
        self.operators.Num_samples = n_samples
        self.num_samples = n_samples
        if save_path is not None:
            self.save_path = save_path

        if not isinstance(names, list):
            names = [names]

        image_dict = {}

        for name in names:
            alfa_path = os.path.join(path, name)
            data_name_list = os.listdir(alfa_path)

            name = name.split(".")[0]

            if is_video:
                image_dict[name] = ft.load_video(alfa_path, data_name_list[0], function=self.operators.reshape)
            else:
                image_dict[name] = (ft.load_image(alfa_path, data_name_list, function=self.operators.reshape))

        for i in range(self.num_samples):
            self.operators.index_sample = i
            ft.save(
                path=save_path,
                data_dict=image_dict,
                function=self.operators.shift,
                aux_num_iter=i
            )

    # def shit(self, save_path, n_samples=5, shape_input=None, is_video=False):
    #
    #     self.operators = OP(self.shift_vecto, n_samples, shape_input)
    #
    #     if is_video:
    #         image_dict = ft.load_video(self.path, self.images_phat, function=self.operators.reshape)
    #     else:
    #         image_dict = ft.load_image(self.path, self.images_phat, function=self.operators.reshape)
    #
    #     for name, image_list in image_dict.items():
    #         aux_save_path = save_path
    #         for index_sample in range(n_samples):
    #             for index, image in enumerate(image_list):
    #                 self.list_save_path.append("{}_{}".format(name, index_sample))
    #                 ft.save(path=aux_save_path,
    #                         dir_images="{}_{}".format(name, index_sample),
    #                         images=self.operators.shift(image, index_sample),
    #                         names="{}_{}".format(name, index))

    def build_image(self, load_path, save_path, names_path: list):
        image_path_dict = {}

        for sub_path in names_path:
            alfa_path = os.path.join(load_path, sub_path)
            image_path_list = os.listdir(alfa_path)
            image = None
            for index, image_path in enumerate(image_path_list):
                aux_path = os.path.join(alfa_path, image_path)
                image_name_list = os.listdir(aux_path)
                print(index)
                self.operators.index_sample = index
                if image is None:
                    image = np.stack(ft.load_image(aux_path, image_name_list, function=self.operators.un_shift)) \
                            / self.operators.Num_samples
                else:
                    image = image + (np.stack(ft.load_image(aux_path, image_name_list, function=self.operators.un_shift))\
                            / self.operators.Num_samples)


            image_path_dict[sub_path] = image

        ft.save(path=save_path,
                data_dict=image_path_dict
        )


