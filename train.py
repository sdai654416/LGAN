import argparse
import os
import sys

import chainer
import pdb
from chainer import training
from chainer.training import extension
from chainer.training import extensions

sys.path.append(os.path.dirname(__file__))

from common.dataset import Cifar10Dataset
from common.evaluation import sample_generate, sample_generate_light, calc_inception, calc_FID, calc_AIS
from common.record import record_setting
import common.net

def make_optimizer(model, alpha, beta1, beta2):
    optimizer = chainer.optimizers.Adam(alpha=alpha, beta1=beta1, beta2=beta2)
    optimizer.setup(model)
    return optimizer

def main():
    parser = argparse.ArgumentParser(description='Train script')
    parser.add_argument('--algorithm', '-a', type=str, default="dcgan", help='GAN algorithm')
    parser.add_argument('--architecture', type=str, default="dcgan", help='Network architecture')
    parser.add_argument('--batchsize', type=int, default=64)
    parser.add_argument('--z_dim', type=int, default=2)
    parser.add_argument('--bound', type=float, default=1.0)
    parser.add_argument('--h_dim', type=int, default=128)
    parser.add_argument('--max_iter', type=int, default=150000)
    parser.add_argument('--gpu', '-g', type=int, default=0, help='GPU ID (negative value indicates CPU)')
    parser.add_argument('--out', '-o', default='result', help='Directory to output the result')
    parser.add_argument('--snapshot_interval', type=int, default=10000, help='Interval of snapshot')
    parser.add_argument('--evaluation_interval', type=int, default=10000, help='Interval of evaluation')
    parser.add_argument('--display_interval', type=int, default=100, help='Interval of displaying log to console')
    parser.add_argument('--n_dis', type=int, default=1, help='number of discriminator update per generator update')
    parser.add_argument('--gamma', type=float, default=0.5, help='hyperparameter gamma')
    parser.add_argument('--lam', type=float, default=10, help='gradient penalty')
    parser.add_argument('--shift_interval', type=float, default=10000, help='Interval of shift annealing')
    parser.add_argument('--adam_alpha', type=float, default=0.0002, help='alpha in Adam optimizer')
    parser.add_argument('--adam_beta1', type=float, default=0.0, help='beta1 in Adam optimizer')
    parser.add_argument('--adam_beta2', type=float, default=0.9, help='beta2 in Adam optimizer')
    parser.add_argument('--output_dim', type=int, default=256, help='output dimension of the discriminator (for cramer GAN)')

    args = parser.parse_args()
    record_setting(args.out)
    report_keys = ["inception_mean", "inception_std", "FID"]
    # report_keys = ["loss_dis", "loss_gen", "ais", "inception_mean", "inception_std", "FID"]

    # Set up dataset
    train_dataset = Cifar10Dataset()
    train_iter = chainer.iterators.SerialIterator(train_dataset, args.batchsize)
    
    # Setup algorithm specific networks and updaters
    models = []
    opts = {}
    updater_args = {
        "iterator": {'main': train_iter},
        "device": args.gpu
    }

    if args.algorithm == "gan":
        from gan.net import Discriminator, Denoiser
        from gan.updater import Updater
        if args.architecture=="dcgan":
            generator = common.net.DCGANGenerator()
            discriminator = Discriminator()
            denoiser = Denoiser()
        else:
            raise NotImplementedError()
        opts["opt_den"] = make_optimizer(denoiser, args.adam_alpha, args.adam_beta1, args.adam_beta2)
        report_keys.append("loss_den")
        models = [generator, discriminator, denoiser]
        updater_args["n_dis"] = args.n_dis
        updater_args["shift_interval"] = args.shift_interval
        updater_args['n_iter'] = args.max_iter
    
    elif args.algorithm == "wgan_gp":
        from wgan_gp.updater import Updater
        if args.architecture=="dcgan":
            generator = common.net.DCGANGenerator()
            discriminator = common.net.WGANDiscriminator()
            denoiser = common.net.Denoiser()
        else:
            raise NotImplementedError()
        opts["opt_den"] = make_optimizer(denoiser, args.adam_alpha, args.adam_beta1, args.adam_beta2)
        report_keys.append("loss_den")
        report_keys.append("loss_gp")
        models = [generator, discriminator, denoiser]
        updater_args["n_dis"] = args.n_dis
        updater_args["lam"] = args.lam
        updater_args["shift_interval"] = args.shift_interval
        updater_args['n_iter'] = args.max_iter
    else:
        raise NotImplementedError()


    if args.gpu >= 0:
        chainer.cuda.get_device_from_id(args.gpu).use()
        print("use gpu {}".format(args.gpu))
        for m in models:
            m.to_gpu()

    # Set up optimizers
    opts["opt_gen"] = make_optimizer(generator, args.adam_alpha, args.adam_beta1, args.adam_beta2)
    opts["opt_dis"] = make_optimizer(discriminator, args.adam_alpha, args.adam_beta1, args.adam_beta2)

    updater_args["optimizer"] = opts
    updater_args["models"] = models

    # Set up updater and trainer
    updater = Updater(**updater_args)
    trainer = training.Trainer(updater, (args.max_iter, 'iteration'), out=args.out)

    # Set up logging
    for m in models:
        trainer.extend(extensions.snapshot_object(
            m, m.__class__.__name__ + '_{.updater.iteration}.npz'), trigger=(args.snapshot_interval, 'iteration'))
    trainer.extend(extensions.LogReport(keys=report_keys,
                                        trigger=(args.display_interval, 'iteration')))
    trainer.extend(extensions.PrintReport(report_keys), trigger=(args.display_interval, 'iteration'))

    trainer.extend(sample_generate(generator, args.out), trigger=(args.evaluation_interval, 'iteration'),
                   priority=extension.PRIORITY_WRITER)
    trainer.extend(sample_generate_light(generator, args.out), trigger=(args.evaluation_interval // 10, 'iteration'),
                   priority=extension.PRIORITY_WRITER)
    trainer.extend(calc_inception(generator), trigger=(args.evaluation_interval, 'iteration'),
                   priority=extension.PRIORITY_WRITER)
    trainer.extend(calc_FID(generator), trigger=(args.evaluation_interval, 'iteration'),
                   priority=extension.PRIORITY_WRITER)
    trainer.extend(extensions.ProgressBar(update_interval=10))

    # Run the training
    trainer.run()


if __name__ == '__main__':
    main()
