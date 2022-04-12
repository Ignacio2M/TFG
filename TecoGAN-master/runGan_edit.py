#!/home/chona/anaconda3/envs/venv-3.7/bin/python
'''
several running examples, run with
python3 runGan.py 1 # the last number is the run case number

runcase == 1    inference a trained model

'''
import json
import os, subprocess, sys, shutil


def preexec():  # Don't forward signals.
    os.setpgrp()


def mycall(cmd, block=False):
    if not block:
        return subprocess.Popen(cmd)
    else:
        return subprocess.Popen(cmd, preexec_fn=preexec)


def folder_check(path):
    try_num = 1
    oripath = path[:-1] if path.endswith('/') else path
    while os.path.exists(path):
        print("Delete existing folder " + path + "?(Y/N)")
        decision = input()
        if decision == "Y":
            shutil.rmtree(path, ignore_errors=True)
            break
        else:
            path = oripath + "_%d/" % try_num
            try_num += 1
            print(path)

    return path


# ======================================================================================================================


with open("../Test_images/info.json", "r") as json_file:
    info = json.load(json_file)["SR_info"]
    uuid = list(info.keys())[-1]
    info = info[uuid]

load_path = "../"+info["Path"]["Save_pre"]
dir_list = info["Path"]["Dir"]

for direct in dir_list:
    dir_patha = "{}/{}".format(load_path, direct)
    list_image_dir = os.listdir(dir_patha)
    save_path = "../"+info["Path"]["Save_net"] + "/" + direct

    if not os.path.exists(save_path):
        os.makedirs(save_path)
    # run these test cases one by one:
    for nn, name_path in enumerate(list_image_dir):


        print(os.path.join(load_path, name_path))
        print("\t {0} Directorio: {1} {0}".format("=" * 10, name_path))
        cmd1 = ["python3", "main.py",
                "--cudaID", "0",  # set the cudaID here to use only one GPU
                "--output_dir", os.path.join(save_path, name_path),  # Set the place to put the results.
                "--summary_dir", os.path.join(save_path, 'log/'),  # Set the place to put the log.
                "--mode", "inference",
                "--input_dir_LR", os.path.join(dir_patha, name_path),  # the LR directory
                # "--input_dir_HR", os.path.join("./HR/", testpre[nn]),  # the HR directory
                # one of (input_dir_HR,input_dir_LR) should be given
                # "--output_pre", name_path,  # the subfolder to save current scene, optional
                "--num_resblock", "16",  # our model has 16 residual blocks,
                # the pre-trained FRVSR and TecoGAN mini have 10 residual blocks
                "--checkpoint", './model/TecoGAN',  # the path of the trained model,
                "--output_ext", "png"  # png is more accurate, jpg is smaller
                ]
        mycall(cmd1).communicate()