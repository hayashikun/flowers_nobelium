import argparse
from os import path

import chainer
from chainer import training
from chainer.training import extensions

import data
import models


def main():
    parser = argparse.ArgumentParser(description="Learning from flowers data")
    parser.add_argument("--gpu", "-g", type=int, default=-1, help="GPU ID (negative value indicates CPU")
    parser.add_argument("--init", help="Initialize the model from given file")
    parser.add_argument('--job', '-j', type=int, help='Number of parallel data loading processes')
    parser.add_argument("--resume", '-r', default='', help="Initialize the trainer from given file")
    args = parser.parse_args()

    batch = 10
    epoch = 100
    val_batch = 200
    model = models.ResNet50V1(data.ClassNumber)
    if args.init:
        print('Load model from', args.initmodel)
        chainer.serializers.load_npz(args.init, model)
    if args.gpu >= 0:
        chainer.backends.cuda.get_device_from_id(args.gpu).use()  # Make the GPU current
        model.to_gpu()

    if data.fetch_flowers() and data.fetch_labels():
        print("Flower images and labels have been fetched.")
    else:
        print("Failed to fetch flower images and labels")
        return

    data.pre_process_data(224)

    output_path = path.join(path.dirname(__file__), "out", model.__class__.__name__)

    train, validate = data.get_datasets()

    train_iter = chainer.iterators.MultiprocessIterator(train, batch, n_processes=args.job)
    val_iter = chainer.iterators.MultiprocessIterator(validate, val_batch, repeat=False, n_processes=args.job)

    classifier = chainer.links.Classifier(model)
    optimizer = chainer.optimizers.Adam()
    optimizer.setup(classifier)
    model.base.disable_update()

    updater = training.updaters.StandardUpdater(train_iter, optimizer, device=args.gpu)
    trainer = training.Trainer(updater, (epoch, 'epoch'), output_path)

    val_interval = 10000, 'iteration'
    log_interval = 100, 'iteration'

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

    print("Start training")
    trainer.run()


if __name__ == '__main__':
    main()
