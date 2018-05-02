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
$ nohup python train_model.py -g 0 > stdout.txt &
```
