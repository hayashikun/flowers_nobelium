import os
from os import path
import urllib.request
import urllib.error
import tarfile
import scipy.io
import pandas as pd


DataPath = path.join(path.dirname(__file__), "data")


def get_flowers():
    img_dir = path.join(DataPath, "flowers")
    if path.isdir(img_dir):
        return True
    tgz_path = path.join(DataPath, "102flowers.tgz")
    if not path.isfile(tgz_path):
        url = "http://www.robots.ox.ac.uk/~vgg/data/flowers/102/102flowers.tgz"
        try:
            urllib.request.urlretrieve(url, tgz_path)
        except urllib.error.URLError:
            return False
    extract_tar(tgz_path, DataPath)
    jpg_path = path.join(DataPath, "jpg")
    if not path.exists(jpg_path):
        return False
    os.rename(jpg_path, img_dir)
    return True


def get_labels():
    labels_path = path.join(DataPath, "labels.csv")
    if path.isdir(labels_path):
        return True
    mat_path = path.join(DataPath, "imagelabels.mat")
    if not path.isfile(mat_path):
        url = "http://www.robots.ox.ac.uk/~vgg/data/flowers/102/imagelabels.mat"
        try:
            urllib.request.urlretrieve(url, mat_path)
        except urllib.error.URLError:
            return False
    mat = scipy.io.loadmat(mat_path)
    labels = mat["labels"][0]
    images = ["image_{:05}.jpg".format(i + 1) for i in range(len(labels))]
    df = pd.DataFrame({"image": images, "label": labels})
    df.to_csv(labels_path)
    return True


def extract_tar(tar_path, extract_path):
    tar = tarfile.open(tar_path, 'r')
    for item in tar:
        tar.extract(item, extract_path)
        if item.name.find(".tgz") != -1 or item.name.find(".tar") != -1:
            extract_tar(item.name, "./" + item.name[:item.name.rfind('/')])