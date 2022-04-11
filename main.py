import subprocess

import numpy as np

import Utils.file_tools as ft
from Transforms import SR

# Press the green button in the gutter to run the script.

dir_images = ["Test_1"]

if __name__ == '__main__':

    sr = SR.SR( init_load_path= ft.test_original_path,
        init_save_path= ft.test_out_pre_path,
        final_load_path= ft.test_net_path,
        final_save_path= ft.test_SR_path,
        rotate_increment= 15,
        translate_vecto= np.array([4,3])
                )

    # sr.init_sr(["Test_1"], 47, take_range=[20,40], angle_space_const=False)
    # subprocess.run("../Tranformaciones/TecoGAN-master/init_test.sh", stdout=True, shell=True)
    sr.build_image(["Test_1"])




