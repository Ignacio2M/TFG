import subprocess
import Utils.file_tools as ft
from Transform import SR


# Press the green button in the gutter to run the script.

dir_images = ["Test_1"]

if __name__ == '__main__':

    sr = SR.SR((0, 2))
    # frame = vt.split_frame()

    sr.shift("./Test_images/Video_Original/", ft.test_out_pre_path, "Test_1", n_samples=47,is_video=True)
    subprocess.run("../Tranformaciones/TecoGAN-master/init_test.sh", stdout=True, shell=True)
    sr.build_image(ft.test_net_path, ft.test_SR_path, dir_images)




