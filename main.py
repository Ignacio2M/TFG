import subprocess
import Utils.file_tools as ft
from Transform import SR

# Press the green button in the gutter to run the script.

dir_images = ['calendar_0', 'calendar_1', 'calendar_2', 'calendar_3', 'calendar_4', 'calendar_5',
              'calendar_6', 'calendar_7', 'calendar_8', 'calendar_9', 'calendar_10', 'calendar_11',
              'calendar_12', 'calendar_13', 'calendar_14', 'calendar_15', 'calendar_16', 'calendar_17',
              'calendar_18', 'calendar_19', 'calendar_20', 'calendar_21', 'calendar_22', 'calendar_23',
              'calendar_24', 'calendar_25', 'calendar_26', 'calendar_27', 'calendar_28', 'calendar_29',
              'calendar_30', 'calendar_31', 'calendar_32', 'calendar_33', 'calendar_34', 'calendar_35',
              'calendar_36', 'calendar_37', 'calendar_38', 'calendar_39', 'calendar_40', 'calendar_41',
              'calendar_42', 'calendar_43', 'calendar_44', 'calendar_45', 'calendar_46']

if __name__ == '__main__':
    sr = SR.SR(ft.test_input_path, (0, 2), ["calendar"])
    sr.shit(ft.test_out_pre_path ,n_samples=47)
    subprocess.run("../Tranformaciones/TecoGAN-master/init_test.sh", stdout=True, shell=True)
    sr.build_image(ft.test_net_path, dir_images)
