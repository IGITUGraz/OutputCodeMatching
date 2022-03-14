import numpy as np


def lr_scheduler(optimizer, epoch, args):
    if args.schedule == 'step':
        if epoch <= 0.5 * args.epochs:
            lr = args.lr
        elif epoch <= 0.75 * args.epochs:
            lr = args.lr * 0.1
        else:
            lr = args.lr * 0.01
    elif args.schedule == 'cosine':
        lr = args.lr * 0.5 * (1 + np.cos((epoch - 1) / args.epochs * np.pi))
    else:
        lr = args.lr

    for param_group in optimizer.param_groups:
        param_group["lr"] = lr

    return lr
