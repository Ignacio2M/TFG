import cv2 as cv
import os
import numpy as np

import Utils.file_tools as ft


class OP:

    def __init__(self, shift_vector, max_num_samples, shape_image):
        self.Shift_vector = shift_vector
        self.Num_samples = max_num_samples
        self.scale_factor = 4
        self.shape = shape_image

    def reshape(self, image):
        image = np.append(image,
                          np.zeros((self.Shift_vector[1] * self.Num_samples, image.shape[1], 3), dtype=image.dtype),
                          axis=0)
        image = np.append(image,
                          np.zeros((image.shape[0], self.Shift_vector[0] * self.Num_samples, 3), dtype=image.dtype),
                          axis=1)
        return image

    def shift(self, image, index_sample):
        image = np.roll(np.roll(image, self.Shift_vector[1] * index_sample, axis=0),
                        self.Shift_vector[0] * index_sample, axis=1)
        return image

    def un_shit(self, image, index):
        init_point = (self.Shift_vector[0] * self.scale_factor * index,
                      self.Shift_vector[1] * self.scale_factor * index)
        final_point = (init_point[0] + (self.shape[0] * self.scale_factor),
                       init_point[1] + (self.shape[1] * self.scale_factor))

        return image[init_point[1]:final_point[1], init_point[0]:final_point[0], ::]


class SR:
    save_path = None
    images_phat: list
    shift_vecto = None
    num_samples = None
    path: str
    operators: OP

    def __init__(self, path, shift_vector, image_phat):
        self.path = path
        self.save_path = path
        self.shift_vecto = shift_vector
        self.images_phat = image_phat
        self.list_save_path = []

    def shit(self, save_path, n_samples=5, shape_input=(180, 144)):

        save_path = save_path
        self.operators = OP(self.shift_vecto, n_samples, shape_input)
        image_dict = ft.load_image(self.path, self.images_phat, function=self.operators.reshape)

        for name, image_list in image_dict.items():
            aux_save_path = save_path
            for index_sample in range(n_samples):
                for index, image in enumerate(image_list):
                    self.list_save_path.append("{}_{}".format(name, index_sample))
                    ft.save(path=aux_save_path,
                            dir_images="{}_{}".format(name, index_sample),
                            images=self.operators.shift(image, index_sample),
                            names="{}_{}".format(name, index))

    def build_image(self, load_path, dir_images):
        image_dict = ft.load_image(load_path, dir_images)
        image_final = {}
        for index, (name, image_list) in enumerate(image_dict.items()):
            for image_index, image in enumerate(image_list):
                image = self.operators.un_shit(image, index)
                img = image_final.get(image_index, [])
                img.append(image)
                image_final[image_index] = img

        for poss, image in image_final.items():
            image = np.array(image, dtype=np.float32)
            image = np.mean(image, axis=0)

            ft.save(path=ft.test_SR_path,
                    dir_images="Out_final",
                    images=image
                    , names=["media{}".format(poss)]
                    )
