import torch
import numpy as np
from torch import nn
from torch import optim
import os
from torchvision import datasets, transforms, models

def data_loader(root, IMG_SIZE, batch_size=256, workers=1, pin_memory=True):
    traindir = os.path.join(root, 'train')
    valdir = os.path.join(root, 'val')
    testdir = os.path.join(root, 'test')
    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                     std=[0.229, 0.224, 0.225])

    train_dataset = datasets.ImageFolder(
        traindir,
        transforms.Compose([
            transforms.Resize(size=(IMG_SIZE, IMG_SIZE)),
            transforms.ToTensor(),
            normalize
        ])
    )
    val_dataset = datasets.ImageFolder(
        valdir,
        transforms.Compose([
            transforms.Resize(size=(IMG_SIZE, IMG_SIZE)),
            transforms.ToTensor(),
            normalize
        ])
    )
    test_dataset = datasets.ImageFolder(
        testdir,
        transforms.Compose([
            transforms.Resize(size=(IMG_SIZE, IMG_SIZE)),
            transforms.ToTensor(),
            normalize
        ])
    )

    train_loader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=workers,
        pin_memory=pin_memory,
        sampler=None
    )
    val_loader = torch.utils.data.DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=workers,
        pin_memory=pin_memory
    )
    test_loader = torch.utils.data.DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=workers,
        pin_memory=pin_memory
    )
    return train_loader, val_loader, test_loader, train_dataset, val_dataset, test_dataset