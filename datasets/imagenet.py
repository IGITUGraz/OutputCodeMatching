import os
import torch
import numpy as np
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
from scipy.linalg import hadamard


class ImageNet:
    def __init__(self, args, normalize=True):
        self.args = args
        self.norm_layer = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        self.tr_train = [transforms.RandomResizedCrop(224),
                         transforms.RandomHorizontalFlip(),
                         transforms.ToTensor()]
        self.tr_test = [transforms.Resize(256),
                        transforms.CenterCrop(224),
                        transforms.ToTensor()]

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
        trainset = datasets.ImageFolder(os.path.join(self.args.data_dir, 'imagenet/train'), self.tr_train,
                                        target_transform=self.tr_target)
        testset = datasets.ImageFolder(os.path.join(self.args.data_dir, 'imagenet/validation'), self.tr_test,
                                       target_transform=self.tr_target)

        train_loader = DataLoader(trainset, shuffle=True, batch_size=self.args.batch,
                                  num_workers=8, pin_memory=True, **kwargs)
        test_loader = DataLoader(testset, batch_size=self.args.batch, shuffle=False,
                                 num_workers=8, pin_memory=True, **kwargs)

        return train_loader, test_loader
