'''
several running examples, run with
python3 runGan.py 1 # the last number is the run case number

runcase == 1    inference a trained model

'''
import os, subprocess, sys, shutil

runcase = int(sys.argv[1])
print("Testing test case %d" % runcase)
input_dir = int(sys.argv[2])
print("Testing test case %d" % runcase)
input_dir = int(sys.argv[2])
print("Testing test case %d" % runcase)

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


if (runcase == 0):  # download inference data, trained models
    # download the trained model
    if (not os.path.exists("./model/")): os.mkdir("./model/")
    cmd1 = "wget https://ge.in.tum.de/download/data/TecoGAN/model.zip -O model/model.zip;"
    cmd1 += "unzip model/model.zip -d model; rm model/model.zip"
    subprocess.call(cmd1, shell=True)

    # download some test data
    cmd2 = "wget https://ge.in.tum.de/download/data/TecoGAN/vid3_LR.zip -O LR/vid3.zip;"
    cmd2 += "unzip LR/vid3.zip -d LR; rm LR/vid3.zip"
    subprocess.call(cmd2, shell=True)

    cmd2 = "wget https://ge.in.tum.de/download/data/TecoGAN/tos_LR.zip -O LR/tos.zip;"
    cmd2 += "unzip LR/tos.zip -d LR; rm LR/tos.zip"
    subprocess.call(cmd2, shell=True)

    # download the ground-truth data
    if (not os.path.exists("./HR/")): os.mkdir("./HR/")
    cmd3 = "wget https://ge.in.tum.de/download/data/TecoGAN/vid4_HR.zip -O HR/vid4.zip;"
    cmd3 += "unzip HR/vid4.zip -d HR; rm HR/vid4.zip"
    subprocess.call(cmd3, shell=True)

    cmd3 = "wget https://ge.in.tum.de/download/data/TecoGAN/tos_HR.zip -O HR/tos.zip;"
    cmd3 += "unzip HR/tos.zip -d HR; rm HR/tos.zip"
    subprocess.call(cmd3, shell=True)

elif (runcase == 1):  # inference a trained model
    dirstr = "./../Test_images/OutPut/Net/Test_1/"  # the place to save the results

    testpre = os.listdir("../Test_images/OutPut/Pre/Test_1/")
    if not os.path.exists(dirstr): os.mkdir(dirstr)

    # run these test cases one by one:
    for nn in range(len(testpre)):
        cmd1 = ["python3", "main.py",
                "--cudaID", "0",  # set the cudaID here to use only one GPU
                "--output_dir", dirstr,  # Set the place to put the results.
                "--summary_dir", os.path.join(dirstr, 'log/'),  # Set the place to put the log.
                "--mode", "inference",
                "--input_dir_LR", os.path.join("../Test_images/OutPut/Pre/Test_1/", testpre[nn]),  # the LR directory
                # "--input_dir_HR", os.path.join("./HR/", testpre[nn]),  # the HR directory
                # one of (input_dir_HR,input_dir_LR) should be given
                "--output_pre", testpre[nn],  # the subfolder to save current scene, optional
                "--num_resblock", "16",  # our model has 16 residual blocks,
                # the pre-trained FRVSR and TecoGAN mini have 10 residual blocks
                "--checkpoint", './model/TecoGAN',  # the path of the trained model,
                "--output_ext", "png"  # png is more accurate, jpg is smaller
                ]
        mycall(cmd1).communicate()
