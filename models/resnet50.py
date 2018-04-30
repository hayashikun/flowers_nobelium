import chainer
from chainer import functions, initializers, links


class BottleNeckA(chainer.Chain):

    def __init__(self, in_size, ch, out_size, stride=2, groups=1):
        super(BottleNeckA, self).__init__()
        w = initializers.HeNormal()

        with self.init_scope():
            self.conv1 = links.Convolution2D(in_size, ch, 1, stride, 0, initialW=w, nobias=True)
            self.bn1 = links.BatchNormalization(ch)
            self.conv2 = links.Convolution2D(ch, ch, 3, 1, 1, initialW=w, nobias=True, groups=groups)
            self.bn2 = links.BatchNormalization(ch)
            self.conv3 = links.Convolution2D(ch, out_size, 1, 1, 0, initialW=w, nobias=True)
            self.bn3 = links.BatchNormalization(out_size)

            self.conv4 = links.Convolution2D(in_size, out_size, 1, stride, 0, initialW=w, nobias=True)
            self.bn4 = links.BatchNormalization(out_size)

    def __call__(self, x):
        h1 = functions.relu(self.bn1(self.conv1(x)))
        h1 = functions.relu(self.bn2(self.conv2(h1)))
        h1 = self.bn3(self.conv3(h1))
        h2 = self.bn4(self.conv4(x))

        return functions.relu(h1 + h2)


class BottleNeckB(chainer.Chain):

    def __init__(self, in_size, ch, groups=1):
        super(BottleNeckB, self).__init__()
        w = initializers.HeNormal()

        with self.init_scope():
            self.conv1 = links.Convolution2D(in_size, ch, 1, 1, 0, initialW=w, nobias=True)
            self.bn1 = links.BatchNormalization(ch)
            self.conv2 = links.Convolution2D(ch, ch, 3, 1, 1, initialW=w, nobias=True, groups=groups)
            self.bn2 = links.BatchNormalization(ch)
            self.conv3 = links.Convolution2D(ch, in_size, 1, 1, 0, initialW=w, nobias=True)
            self.bn3 = links.BatchNormalization(in_size)

    def __call__(self, x):
        h = functions.relu(self.bn1(self.conv1(x)))
        h = functions.relu(self.bn2(self.conv2(h)))
        h = self.bn3(self.conv3(h))

        return functions.relu(h + x)


class Block(chainer.ChainList):

    def __init__(self, layer, in_size, ch, out_size, stride=2, groups=1):
        super(Block, self).__init__()
        self.add_link(BottleNeckA(in_size, ch, out_size, stride, groups))
        for i in range(layer - 1):
            self.add_link(BottleNeckB(out_size, ch, groups))

    def __call__(self, x):
        for f in self.children():
            x = f(x)
        return x


class ResNet50(chainer.Chain):
    in_size = 224

    def __init__(self):
        super(ResNet50, self).__init__()
        with self.init_scope():
            self.conv1 = links.Convolution2D(3, 64, 7, 2, 3, initialW=initializers.HeNormal())
            self.bn1 = links.BatchNormalization(64)
            self.res2 = Block(3, 64, 64, 256, 1)
            self.res3 = Block(4, 256, 128, 512)
            self.res4 = Block(6, 512, 256, 1024)
            self.res5 = Block(3, 1024, 512, 2048)
            self.fc = links.Linear(2048, 1000)

    def __call__(self, x, t):
        h = self.bn1(self.conv1(x))
        h = functions.max_pooling_2d(functions.relu(h), 3, stride=2)
        h = self.res2(h)
        h = self.res3(h)
        h = self.res4(h)
        h = self.res5(h)
        h = functions.average_pooling_2d(h, 7, stride=1)
        h = self.fc(h)

        loss = functions.softmax_cross_entropy(h, t)
        chainer.report({'loss': loss, 'accuracy': functions.accuracy(h, t)}, self)
        return loss


class ResNeXt50(ResNet50):
    in_size = 224

    def __init__(self):
        chainer.Chain.__init__(self)
        with self.init_scope():
            self.conv1 = links.Convolution2D(3, 64, 7, 2, 3, initialW=initializers.HeNormal())
            self.bn1 = links.BatchNormalization(64)
            self.res2 = Block(3, 64, 128, 256, 1, groups=32)
            self.res3 = Block(4, 256, 256, 512, groups=32)
            self.res4 = Block(6, 512, 512, 1024, groups=32)
            self.res5 = Block(3, 1024, 1024, 2048, groups=32)
            self.fc = links.Linear(2048, 1000)
