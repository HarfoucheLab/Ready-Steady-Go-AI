from collections import OrderedDict
import torch
import numpy as np
from torch import nn
from torch import optim
import os
from torchvision import datasets, transforms, models
import itertools
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix

from ..dcnnModelBuilder import *
from ..tools import *
from ..dataloaders import *

def test_get_all_preds():
    DENSENET_PRETRAINED_PATH ='/content/models/RSGAI_DenseNet.pth'
    model = loadModelWeights(DENSENET_PRETRAINED_PATH)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    data_dir = '/content/dataset/tomato-dataset/'
    IMG_SIZE = 220
    NUM_WORKERS = 1
    BATCH_SIZE = 20

    train_loader, val_loader, test_loader, train_dataset, val_dataset, test_dataset = data_loader(data_dir, IMG_SIZE, BATCH_SIZE, NUM_WORKERS, False)

    model.eval()
    test_preds, test_labels = get_all_preds(model, test_loader)
    assert len(test_preds) == 1825

    
def test_get_num_correct():
    
    DENSENET_PRETRAINED_PATH ='/content/models/RSGAI_DenseNet.pth'
    model = loadModelWeights(DENSENET_PRETRAINED_PATH)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    data_dir = '/content/dataset/tomato-dataset/'
    IMG_SIZE = 220
    NUM_WORKERS = 1
    BATCH_SIZE = 20

    train_loader, val_loader, test_loader, train_dataset, val_dataset, test_dataset = data_loader(data_dir, IMG_SIZE, BATCH_SIZE, NUM_WORKERS, False)

    model.eval()
    test_preds, test_labels = get_all_preds(model, test_loader)
    preds_correct = get_num_correct(test_preds.to(device), test_labels.to(device))
    assert preds_correct == 1770