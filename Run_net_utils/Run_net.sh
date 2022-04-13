#! /bin/bash
source ~/anaconda3/etc/profile.d/conda.sh
conda activate venv-3.7
cd ./TecoGAN-master
python "./run_net.py"