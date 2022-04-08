# https://www.kaggle.com/datasets/jessicali9530/celeba-dataset

import os
import argparse
import zipfile

parser = argparse.ArgumentParser()
parser.add_argument("--data_dir", type = str, help = "zip filename")
parser.add_argument("--result_dir", type = str, help = "dir images will be saved")
parser = parser.parse_args()

data_dir = parser.data_dir
result_dir = parser.result_dir

with zipfile.ZipFile(data_dir, 'r') as zip_ref:
    zip_ref.extractall(result_dir)