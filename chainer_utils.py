import os
import urllib.error
import urllib.request

DatasetRoot = os.environ.get('CHAINER_DATASET_ROOT', os.path.expanduser('~/.chainer/dataset'))
PreTrainedModelsDirectory = os.path.join(DatasetRoot, "pfnet", "chainer", "models")


def download_pre_trained_caffemodel(url, path):
    if os.path.exists(path):
        return True
    try:
        urllib.request.urlretrieve(url, path)
    except urllib.error.URLError:
        return False
    return True
