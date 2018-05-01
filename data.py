import os
import shutil
import tarfile
import urllib.error
import urllib.request
from os import path

import numpy as np
import pandas as pd
import scipy.io
from PIL import Image
from chainer import datasets
from tqdm import tqdm

DataPath = path.join(path.dirname(__file__), "data")
FlowerImagesDirectory = path.join(DataPath, "flowers")
PreProcessedFlowerImagesDirectory = path.join(DataPath, "processed_flowers")
LabelsPath = path.join(DataPath, "labels.csv")
MeanPath = path.join(DataPath, "mean.npy")
SplitDatasetSeed = 0
ClassNumber = 102


def fetch_flowers():
    if path.isdir(FlowerImagesDirectory):
        return True
    if not path.exists(DataPath):
        os.mkdir(DataPath)
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
    os.rename(jpg_path, FlowerImagesDirectory)
    return True


def fetch_labels():
    if path.isdir(LabelsPath):
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
    df.to_csv(LabelsPath)
    return True


def extract_tar(tar_path, extract_path):
    tar = tarfile.open(tar_path, 'r')
    for item in tar:
        tar.extract(item, extract_path)
        if item.name.find(".tgz") != -1 or item.name.find(".tar") != -1:
            extract_tar(item.name, "./" + item.name[:item.name.rfind('/')])


def pre_process_data(image_size):
    pre_process_log_path = path.join(DataPath, "pre_process.log")
    with open(path.join(pre_process_log_path)) as f:
        size = f.read()
        try:
            if image_size == int(size):
                return True
        except ValueError:
            pass
    shutil.rmtree(PreProcessedFlowerImagesDirectory)
    os.mkdir(PreProcessedFlowerImagesDirectory)
    os.remove(MeanPath)
    for f in tqdm(os.listdir(FlowerImagesDirectory)):
        img = Image.open(path.join(FlowerImagesDirectory, f))
        crop_size = min(img.width, img.height)
        img = img.crop(((img.width - crop_size) // 2, (img.height - crop_size) // 2,
                        (img.width + crop_size) // 2, (img.height + crop_size) // 2))
        img = img.resize((image_size, image_size))
        img.save(path.join(PreProcessedFlowerImagesDirectory, f))
    calc_mean()
    with open(pre_process_log_path, "w") as f:
        f.write("{}".format(image_size))
    return True


def get_datasets():
    labels = pd.read_csv(LabelsPath, index_col=0)
    # label: 1 -> 102
    ds = datasets.LabeledImageDataset(list(zip(labels["image"], labels["label"] - 1)), PreProcessedFlowerImagesDirectory)
    return datasets.split_dataset_random(ds, int(len(ds) * 0.8), seed=SplitDatasetSeed)


def calc_mean():
    if os.path.exists(MeanPath):
        return np.load(MeanPath)
    train, _ = get_datasets()

    mean = np.mean([img[:3] for img, _ in train], axis=0)

    np.save(MeanPath, mean)
    return mean
