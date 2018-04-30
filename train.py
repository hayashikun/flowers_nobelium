import argparse
from os import path

import chainer
from chainer import training
from chainer.training import extensions

import data
import models


def main():
    architectures = {
        "alex": models.Alex,
        "alex_fp16": models.AlexFp16,
        "googlenet": models.GoogLeNet,
        "googlenetbn": models.GoogLeNetBN,
        "googlenetbn_fp16": models.GoogLeNetBNFp16,
        "nin": models.NIN,
        "resnet50": models.ResNet50,
        "resnext50": models.ResNeXt50,
    }

    parser = argparse.ArgumentParser(description="Learning from flowers data")
    parser.add_argument("architecture", choices=architectures.keys(), help="Model architecture")
    parser.add_argument("--batch", "-B", type=int, default=32, help="Learning mini-batch size")
    parser.add_argument("--epoch", "-E", type=int, default=100, help="Number of epochs to train")
    parser.add_argument("--gpu", "-g", type=int, default=-1, help="GPU ID (negative value indicates CPU")
    parser.add_argument("--init", help="Initialize the model from given file")
    parser.add_argument('--job', '-j', type=int, help='Number of parallel data loading processes')
    parser.add_argument("--resume", '-r', default='', help="Initialize the trainer from given file")
    parser.add_argument("--val_batch", '-b', type=int, default=200, help="Validation mini-batch size")
    parser.add_argument('--test', action='store_true')
    parser.set_defaults(test=False)
    args = parser.parse_args()

    model = architectures.get(args.architecture)
    if model is None:
        print("Please specify a valid model")
        return
    model = model()
    if args.init:
        print('Load model from', args.initmodel)
        chainer.serializers.load_npz(args.initmodel, model)
    if args.gpu >= 0:
        chainer.backends.cuda.get_device_from_id(
            args.gpu).use()  # Make the GPU current
        model.to_gpu()

    if data.fetch_flowers() and data.fetch_labels():
        print("Flower images and labels have been fetched.")
    else:
        print("Failed to fetch flower images and labels")
        return

    output_path = path.join(path.dirname(__file__), "out", args.architecture)

    train, validate = data.get_datasets()

    train_iter = chainer.iterators.MultiprocessIterator(train, args.batch, n_processes=args.job)
    val_iter = chainer.iterators.MultiprocessIterator(validate, args.val_batch, repeat=False, n_processes=args.job)

    optimizer = chainer.optimizers.MomentumSGD(lr=0.01, momentum=0.9)
    optimizer.setup(model)

    updater = training.updaters.StandardUpdater(train_iter, optimizer, device=args.gpu)
    trainer = training.Trainer(updater, (args.epoch, 'epoch'), output_path)

    val_interval = (1 if args.test else 10000), 'iteration'
    log_interval = (1 if args.test else 100), 'iteration'

    trainer.extend(extensions.Evaluator(val_iter, model, device=args.gpu), trigger=val_interval)
    trainer.extend(extensions.dump_graph('main/loss'))
    trainer.extend(extensions.snapshot(), trigger=val_interval)
    trainer.extend(extensions.snapshot_object(model, 'model_iter_{.updater.iteration}'), trigger=val_interval)

    trainer.extend(extensions.LogReport(trigger=log_interval))
    trainer.extend(extensions.observe_lr(), trigger=log_interval)
    trainer.extend(extensions.PrintReport([
        'epoch', 'iteration', 'main/loss', 'validation/main/loss',
        'main/accuracy', 'validation/main/accuracy', 'lr'
    ]), trigger=log_interval)
    trainer.extend(extensions.ProgressBar(update_interval=10))

    if args.resume:
        chainer.serializers.load_npz(args.resume, trainer)

    trainer.run()


if __name__ == '__main__':
    main()
