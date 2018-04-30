import chainer
from chainer import functions, initializers, links


class NIN(chainer.Chain):
    """Network-in-Network example model."""

    in_size = 227

    def __init__(self):
        super(NIN, self).__init__()
        conv_init = initializers.HeNormal()  # MSRA scaling

        with self.init_scope():
            self.mlpconv1 = links.MLPConvolution2D(None, (96, 96, 96), 11, stride=4, conv_init=conv_init)
            self.mlpconv2 = links.MLPConvolution2D(None, (256, 256, 256), 5, pad=2, conv_init=conv_init)
            self.mlpconv3 = links.MLPConvolution2D(None, (384, 384, 384), 3, pad=1, conv_init=conv_init)
            self.mlpconv4 = links.MLPConvolution2D(None, (1024, 1024, 1000), 3, pad=1, conv_init=conv_init)

    def __call__(self, x, t):
        h = functions.max_pooling_2d(functions.relu(self.mlpconv1(x)), 3, stride=2)
        h = functions.max_pooling_2d(functions.relu(self.mlpconv2(h)), 3, stride=2)
        h = functions.max_pooling_2d(functions.relu(self.mlpconv3(h)), 3, stride=2)
        h = self.mlpconv4(functions.dropout(h))
        h = functions.reshape(functions.average_pooling_2d(h, 6), (len(x), 1000))

        loss = functions.softmax_cross_entropy(h, t)
        chainer.report({'loss': loss, 'accuracy': functions.accuracy(h, t)}, self)
        return loss
