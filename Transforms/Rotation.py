# import numpy as np
# import cv2 as cv
# import os
#
# class Rotation:
#     init_load_path: str
#     init_save_path: str
#     final_load_path: str
#     final_save_path: str
#     rotation_angle: int
#     backGroundImage: np.array
#
#     def __init__(self, init_load_path, init_save_path, final_load_path, final_save_path, rotation_angle) -> None:
#         self.rotation_angle = rotation_angle
#         self.final_save_path = final_save_path
#         self.final_load_path = final_load_path
#         self.init_save_path = init_save_path
#         self.init_load_path = init_load_path
#
#     def _rotateImage(self, image, angle):
#         center = np.array(image.shape[0:3])/3
#         matrix_rotate = cv.getRotationMatrix2D()
#
#
#     def rotate(self, images_path:list, n_samples:int, time_cons=False, function=None):
#         images = list(map())
