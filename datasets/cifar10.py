import os
import torch
import numpy as np
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
from scipy.linalg import hadamard


class CIFAR10:
    def __init__(self, args, normalize=True):
        self.args = args
        self.norm_layer = transforms.Normalize(mean=[0.4914, 0.4822, 0.4465], std=[0.247, 0.243, 0.261])
        self.tr_train = [transforms.RandomCrop(32, padding=4),
                         transforms.RandomHorizontalFlip(),
                         transforms.ToTensor()]
        self.tr_test = [transforms.ToTensor()]

        if normalize:
            self.tr_train.append(self.norm_layer)
            self.tr_test.append(self.norm_layer)

        self.tr_train = transforms.Compose(self.tr_train)
        self.tr_test = transforms.Compose(self.tr_test)

        if args.ocm:
            self.C = hadamard(args.code_length).astype(np.float32)
            self.C = np.delete(self.C, 0, axis=0)
            np.random.shuffle(self.C)
            self.C = self.C[:args.num_classes]
            print(self.C)
            self.tr_target = [transforms.Lambda(lambda y: torch.LongTensor(self.C[y]))]
        else:
            self.tr_target = [transforms.Lambda(lambda y: y)]

        self.tr_target = transforms.Compose(self.tr_target)

    def loaders(self, **kwargs):
        trainset = datasets.CIFAR10(root=os.path.join(self.args.data_dir, 'CIFAR10'), train=True, download=True,
                                    transform=self.tr_train, target_transform=self.tr_target)
        testset = datasets.CIFAR10(root=os.path.join(self.args.data_dir, 'CIFAR10'), train=False, download=True,
                                   transform=self.tr_test, target_transform=self.tr_target)

        train_loader = DataLoader(trainset, batch_size=self.args.batch, shuffle=True, **kwargs)
        test_loader = DataLoader(testset, batch_size=self.args.batch, shuffle=False, **kwargs)

        return train_loader, test_loader
