import chainer
import numpy as np
from chainer import functions
from chainer import initializers
from chainer import links


class GoogLeNetBN(chainer.Chain):
    """New GoogLeNet of BatchNormalization version."""

    in_size = 224

    def __init__(self):
        super(GoogLeNetBN, self).__init__()
        with self.init_scope():
            self.conv1 = links.Convolution2D(
                None, 64, 7, stride=2, pad=3, nobias=True)
            self.norm1 = links.BatchNormalization(64)
            self.conv2 = links.Convolution2D(None, 192, 3, pad=1, nobias=True)
            self.norm2 = links.BatchNormalization(192)
            self.inc3a = links.InceptionBN(None, 64, 64, 64, 64, 96, 'avg', 32)
            self.inc3b = links.InceptionBN(None, 64, 64, 96, 64, 96, 'avg', 64)
            self.inc3c = links.InceptionBN(None, 0, 128, 160, 64, 96, 'max', stride=2)
            self.inc4a = links.InceptionBN(None, 224, 64, 96, 96, 128, 'avg', 128)
            self.inc4b = links.InceptionBN(None, 192, 96, 128, 96, 128, 'avg', 128)
            self.inc4c = links.InceptionBN(None, 160, 128, 160, 128, 160, 'avg', 128)
            self.inc4d = links.InceptionBN(None, 96, 128, 192, 160, 192, 'avg', 128)
            self.inc4e = links.InceptionBN(None, 0, 128, 192, 192, 256, 'max', stride=2)
            self.inc5a = links.InceptionBN(None, 352, 192, 320, 160, 224, 'avg', 128)
            self.inc5b = links.InceptionBN(None, 352, 192, 320, 192, 224, 'max', 128)
            self.out = links.Linear(None, 1000)

            self.conva = links.Convolution2D(None, 128, 1, nobias=True)
            self.norma = links.BatchNormalization(128)
            self.lina = links.Linear(None, 1024, nobias=True)
            self.norma2 = links.BatchNormalization(1024)
            self.outa = links.Linear(None, 1000)

            self.convb = links.Convolution2D(None, 128, 1, nobias=True)
            self.normb = links.BatchNormalization(128)
            self.linb = links.Linear(None, 1024, nobias=True)
            self.normb2 = links.BatchNormalization(1024)
            self.outb = links.Linear(None, 1000)

    def __call__(self, x, t):
        h = functions.max_pooling_2d(functions.relu(self.norm1(self.conv1(x))), 3, stride=2, pad=1)
        h = functions.max_pooling_2d(functions.relu(self.norm2(self.conv2(h))), 3, stride=2, pad=1)

        h = self.inc3a(h)
        h = self.inc3b(h)
        h = self.inc3c(h)
        h = self.inc4a(h)

        a = functions.average_pooling_2d(h, 5, stride=3)
        a = functions.relu(self.norma(self.conva(a)))
        a = functions.relu(self.norma2(self.lina(a)))
        a = self.outa(a)
        loss1 = functions.softmax_cross_entropy(a, t)

        h = self.inc4b(h)
        h = self.inc4c(h)
        h = self.inc4d(h)

        b = functions.average_pooling_2d(h, 5, stride=3)
        b = functions.relu(self.normb(self.convb(b)))
        b = functions.relu(self.normb2(self.linb(b)))
        b = self.outb(b)
        loss2 = functions.softmax_cross_entropy(b, t)

        h = self.inc4e(h)
        h = self.inc5a(h)
        h = functions.average_pooling_2d(self.inc5b(h), 7)
        h = self.out(h)
        loss3 = functions.softmax_cross_entropy(h, t)

        loss = 0.3 * (loss1 + loss2) + loss3
        accuracy = functions.accuracy(h, t)

        chainer.report({
            'loss': loss,
            'loss1': loss1,
            'loss2': loss2,
            'loss3': loss3,
            'accuracy': accuracy,
        }, self)
        return loss


class GoogLeNetBNFp16(GoogLeNetBN):
    """New GoogLeNet of BatchNormalization version."""

    in_size = 224
    dtype = np.float16

    def __init__(self):
        w = initializers.HeNormal(1 / np.sqrt(2), self.dtype)
        bias = initializers.Zero(self.dtype)

        chainer.Chain.__init__(self)
        with self.init_scope():
            dtype: np.dtype = self.dtype
            self.conv1 = links.Convolution2D(None, 64, 7, stride=2, pad=3, initialW=w, nobias=True)
            self.norm1 = links.BatchNormalization(64, dtype=dtype)
            self.conv2 = links.Convolution2D(None, 192, 3, pad=1, initialW=w, nobias=True)
            self.norm2 = links.BatchNormalization(192, dtype=dtype)
            self.inc3a = links.InceptionBN(None, 64, 64, 64, 64, 96, 'avg', 32, conv_init=w, dtype=dtype)
            self.inc3b = links.InceptionBN(None, 64, 64, 96, 64, 96, 'avg', 64, conv_init=w, dtype=dtype)
            self.inc3c = links.InceptionBN(None, 0, 128, 160, 64, 96, 'max', stride=2, conv_init=w, dtype=dtype)
            self.inc4a = links.InceptionBN(None, 224, 64, 96, 96, 128, 'avg', 128, conv_init=w, dtype=dtype)
            self.inc4b = links.InceptionBN(None, 192, 96, 128, 96, 128, 'avg', 128, conv_init=w, dtype=dtype)
            self.inc4c = links.InceptionBN(None, 128, 128, 160, 128, 160, 'avg', 128, conv_init=w, dtype=dtype)
            self.inc4d = links.InceptionBN(None, 64, 128, 192, 160, 192, 'avg', 128, conv_init=w, dtype=dtype)
            self.inc4e = links.InceptionBN(None, 0, 128, 192, 192, 256, 'max', stride=2, conv_init=w, dtype=dtype)
            self.inc5a = links.InceptionBN(None, 352, 192, 320, 160, 224, 'avg', 128, conv_init=w, dtype=dtype)
            self.inc5b = links.InceptionBN(None, 352, 192, 320, 192, 224, 'max', 128, conv_init=w, dtype=dtype)
            self.out = links.Linear(None, 1000, initialW=w, initial_bias=bias)

            self.conva = links.Convolution2D(None, 128, 1, initialW=w, nobias=True)
            self.norma = links.BatchNormalization(128, dtype=dtype)
            self.lina = links.Linear(None, 1024, initialW=w, nobias=True)
            self.norma2 = links.BatchNormalization(1024, dtype=dtype)
            self.outa = links.Linear(None, 1000, initialW=w, initial_bias=bias)

            self.convb = links.Convolution2D(None, 128, 1, initialW=w, nobias=True)
            self.normb = links.BatchNormalization(128, dtype=dtype)
            self.linb = links.Linear(None, 1024, initialW=w, nobias=True)
            self.normb2 = links.BatchNormalization(1024, dtype=dtype)
            self.outb = links.Linear(None, 1000, initialW=w, initial_bias=bias)

    def __call__(self, x, t):
        return GoogLeNetBN.__call__(self, functions.cast(x, self.dtype), t)
