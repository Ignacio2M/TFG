
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


with open("../Test_images/info.json", "r") as json_file:
    info = json.load(json_file)["SR_info"]
    uuid = list(info.keys())[-1]
    info = info[uuid]

dirstr = "../" + info["Path"]["Save_final"] + "/"  # images result
tarstr = "../Test_images/Original/HR/"  # Originals

save_dir = "../" + info["Path"]["Save_metric"] + "/"  # Save path

testpre = info["Path"]["Dir"]  # just put more scenes to evaluate all of them


tar_list = [(tarstr + _) for _ in testpre]
out_list = [(dirstr + _) for _ in testpre]
cmd1 = ["python3", "metrics.py",
        "--output", save_dir,
        "--results", ",".join(out_list),
        "--targets", ",".join(tar_list),
        ]
mycall(cmd1).communicate()
