import torch
import numpy as np
from torch import nn
from torch import optim
import os
from torchvision import datasets, transforms, models

from ..dataloaders import *

def test_data_loader():

    data_dir = '/content/dataset/tomato-dataset/'
    IMG_SIZE = 220
    NUM_WORKERS = 1
    BATCH_SIZE = 20

    train_loader, val_loader, test_loader, train_dataset, val_dataset, test_dataset = data_loader(data_dir, IMG_SIZE, BATCH_SIZE, NUM_WORKERS, False)
    assert len(train_loader.dataset) == 15228
    assert len(val_loader.dataset) == 1812
    assert len(test_loader.dataset) == 1825