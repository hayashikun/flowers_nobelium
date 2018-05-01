import os

import chainer
from chainer import links

import chainer_utils


class ResNet50V1(chainer.Chain):
    def __init__(self, class_labels):
        super(ResNet50V1, self).__init__()
        self.fetch_model()

        with self.init_scope():
            self.base = links.ResNet50Layers()
            self.fc6 = links.Linear(None, class_labels)

    def __call__(self, x):
        h = self.base(x, layers=['pool5'])['pool5']
        return self.fc6(h)

    @staticmethod
    def fetch_model():
        return chainer_utils.download_pre_trained_caffemodel(
            "https://s3-ap-northeast-1.amazonaws.com/hayashikun/ResNet-50-model.caffemodel",
            os.path.join(chainer_utils.PreTrainedModelsDirectory, "ResNet-50-model.caffemodel")
        )
