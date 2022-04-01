dirstr = './results/'  # the place to save the results
sub_dit = []
testpre = ['sample_0', 'sample_1', 'sample_2', 'sample_3', 'sample_4']  # the test cases

import os, subprocess, sys, datetime, signal, shutil

if sub_dit is not None:
    testpre = []
    for dir in sub_dit:
        testpre.extend()


def preexec():  # Don't forward signals.
    os.setpgrp()


def mycall(cmd, block=False):
    if not block:
        return subprocess.Popen(cmd)
    else:
        return subprocess.Popen(cmd, preexec_fn=preexec)


if (not os.path.exists(dirstr)): os.mkdir(dirstr)

# run these test cases one by one:
for nn in range(len(testpre)):
    cmd1 = ["python3", "main.py",
            "--cudaID", "0",  # set the cudaID here to use only one GPU
            "--output_dir", dirstr,  # Set the place to put the results.
            "--summary_dir", os.path.join(dirstr, 'log/'),  # Set the place to put the log.
            "--mode", "inference",
            "--input_dir_LR", os.path.join("./LR/Test/", testpre[nn]),  # the LR directory
            # "--input_dir_HR", os.path.join("./HR/", testpre[nn]),  # the HR directory
            # one of (input_dir_HR,input_dir_LR) should be given
            "--output_pre", testpre[nn],  # the subfolder to save current scene, optional
            "--num_resblock", "16",  # our model has 16 residual blocks,
            # the pre-trained FRVSR and TecoGAN mini have 10 residual blocks
            "--checkpoint", './model/TecoGAN',  # the path of the trained model,
            "--output_ext", "png"  # png is more accurate, jpg is smaller
            ]
    mycall(cmd1).communicate()
