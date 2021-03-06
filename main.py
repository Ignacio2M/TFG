import datetime
import os.path
import subprocess
import sys
import time
import uuid

import numpy as np

import Utils.file_tools as ft
from Transforms import SR

# Press the green button in the gutter to run the script.

dir_images = ["Test_1"]

if __name__ == '__main__':
    sr = SR.SR(uuid="Test_2(2_3_4)_47",
               # uuid=datetime.datetime.now().strftime("%Y_%m_%d_%H_%M_%S"),
               init_load_path=os.path.join(ft.test_original_path, "LR"),
               init_save_path=ft.test_out_pre_path,
               final_load_path=ft.test_net_path,
               final_save_path=ft.test_SR_path,
               info_metric_path=ft.test_metric_path,
               rotate_increment=2,
               translate_vecto=np.array([3, 4])
               )

    sr.init_sr(["Test_2"], 47, angle_space_const=False)
    subprocess.run("./Run_net_utils/Run_net.sh", stdout=sys.stdout, shell=True)
    sr.build_image()
    subprocess.run("./Run_net_utils/Run_test_net.sh", stdout=sys.stdout, shell=True)
