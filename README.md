# Flowers Nobelium

## Train
```
$ mkdir ~/.aws
$ vim ~/.aws/credentials
$ ssh-keygen -t rsa
$ git clone git@github.com:hayashikun/flowers_nobelium.git
$ pip install --upgrade pip
$ pip install -r requirements.txt
$ mkdir -p /home/ubuntu/.chainer/dataset/pfnet/chainer/models/
$ python train_model.py
```
