import chainer
from chainer import functions, links


class GoogLeNet(chainer.Chain):
    in_size = 224

    def __init__(self):
        super(GoogLeNet, self).__init__()
        with self.init_scope():
            self.conv1 = links.Convolution2D(None, 64, 7, stride=2, pad=3)
            self.conv2_reduce = links.Convolution2D(None, 64, 1)
            self.conv2 = links.Convolution2D(None, 192, 3, stride=1, pad=1)
            self.inc3a = links.Inception(None, 64, 96, 128, 16, 32, 32)
            self.inc3b = links.Inception(None, 128, 128, 192, 32, 96, 64)
            self.inc4a = links.Inception(None, 192, 96, 208, 16, 48, 64)
            self.inc4b = links.Inception(None, 160, 112, 224, 24, 64, 64)
            self.inc4c = links.Inception(None, 128, 128, 256, 24, 64, 64)
            self.inc4d = links.Inception(None, 112, 144, 288, 32, 64, 64)
            self.inc4e = links.Inception(None, 256, 160, 320, 32, 128, 128)
            self.inc5a = links.Inception(None, 256, 160, 320, 32, 128, 128)
            self.inc5b = links.Inception(None, 384, 192, 384, 48, 128, 128)
            self.loss3_fc = links.Linear(None, 1000)

            self.loss1_conv = links.Convolution2D(None, 128, 1)
            self.loss1_fc1 = links.Linear(None, 1024)
            self.loss1_fc2 = links.Linear(None, 1000)

            self.loss2_conv = links.Convolution2D(None, 128, 1)
            self.loss2_fc1 = links.Linear(None, 1024)
            self.loss2_fc2 = links.Linear(None, 1000)

    def __call__(self, x, t):
        h = functions.relu(self.conv1(x))
        h = functions.local_response_normalization(functions.max_pooling_2d(h, 3, stride=2), n=5)
        h = functions.relu(self.conv2_reduce(h))
        h = functions.relu(self.conv2(h))
        h = functions.max_pooling_2d(functions.local_response_normalization(h, n=5), 3, stride=2)

        h = self.inc3a(h)
        h = self.inc3b(h)
        h = functions.max_pooling_2d(h, 3, stride=2)
        h = self.inc4a(h)

        ls = functions.average_pooling_2d(h, 5, stride=3)
        ls = functions.relu(self.loss1_conv(ls))
        ls = functions.relu(self.loss1_fc1(ls))
        ls = self.loss1_fc2(ls)
        loss1 = functions.softmax_cross_entropy(ls, t)

        h = self.inc4b(h)
        h = self.inc4c(h)
        h = self.inc4d(h)

        ls = functions.average_pooling_2d(h, 5, stride=3)
        ls = functions.relu(self.loss2_conv(ls))
        ls = functions.relu(self.loss2_fc1(ls))
        ls = self.loss2_fc2(ls)
        loss2 = functions.softmax_cross_entropy(ls, t)

        h = self.inc4e(h)
        h = functions.max_pooling_2d(h, 3, stride=2)
        h = self.inc5a(h)
        h = self.inc5b(h)

        h = functions.average_pooling_2d(h, 7, stride=1)
        h = self.loss3_fc(functions.dropout(h, 0.4))
        loss3 = functions.softmax_cross_entropy(h, t)

        loss = 0.3 * (loss1 + loss2) + loss3
        accuracy = functions.accuracy(h, t)

        chainer.report({
            'loss': loss,
            'loss1': loss1,
            'loss2': loss2,
            'loss3': loss3,
            'accuracy': accuracy
        }, self)
        return loss
