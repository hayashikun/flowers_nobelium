import chainer
import numpy as np
from chainer import functions, initializers, links


class Alex(chainer.Chain):
    """Single-GPU AlexNet without partition toward the channel axis."""

    in_size = 227

    def __init__(self):
        super(Alex, self).__init__()
        with self.init_scope():
            self.conv1 = links.Convolution2D(None, 96, 11, stride=4)
            self.conv2 = links.Convolution2D(None, 256, 5, pad=2)
            self.conv3 = links.Convolution2D(None, 384, 3, pad=1)
            self.conv4 = links.Convolution2D(None, 384, 3, pad=1)
            self.conv5 = links.Convolution2D(None, 256, 3, pad=1)
            self.fc6 = links.Linear(None, 4096)
            self.fc7 = links.Linear(None, 4096)
            self.fc8 = links.Linear(None, 1000)

    def __call__(self, x, t):
        h = functions.max_pooling_2d(functions.local_response_normalization(
            functions.relu(self.conv1(x))), 3, stride=2)
        h = functions.max_pooling_2d(functions.local_response_normalization(
            functions.relu(self.conv2(h))), 3, stride=2)
        h = functions.relu(self.conv3(h))
        h = functions.relu(self.conv4(h))
        h = functions.max_pooling_2d(functions.relu(self.conv5(h)), 3, stride=2)
        h = functions.dropout(functions.relu(self.fc6(h)))
        h = functions.dropout(functions.relu(self.fc7(h)))
        h = self.fc8(h)

        loss = functions.softmax_cross_entropy(h, t)
        chainer.report({'loss': loss, 'accuracy': functions.accuracy(h, t)}, self)
        return loss


class AlexFp16(Alex):
    """Single-GPU AlexNet without partition toward the channel axis."""

    in_size = 227

    def __init__(self):
        chainer.Chain.__init__(self)
        self.dtype = np.float16
        w = initializers.HeNormal(1 / np.sqrt(2), self.dtype)
        bias = initializers.Zero(self.dtype)

        with self.init_scope():
            self.conv1 = links.Convolution2D(None, 96, 11, stride=4, initialW=w, initial_bias=bias)
            self.conv2 = links.Convolution2D(None, 256, 5, pad=2, initialW=w, initial_bias=bias)
            self.conv3 = links.Convolution2D(None, 384, 3, pad=1, initialW=w, initial_bias=bias)
            self.conv4 = links.Convolution2D(None, 384, 3, pad=1, initialW=w, initial_bias=bias)
            self.conv5 = links.Convolution2D(None, 256, 3, pad=1, initialW=w, initial_bias=bias)
            self.fc6 = links.Linear(None, 4096, initialW=w, initial_bias=bias)
            self.fc7 = links.Linear(None, 4096, initialW=w, initial_bias=bias)
            self.fc8 = links.Linear(None, 1000, initialW=w, initial_bias=bias)

    def __call__(self, x, t):
        return Alex.__call__(self, functions.cast(x, self.dtype), t)
