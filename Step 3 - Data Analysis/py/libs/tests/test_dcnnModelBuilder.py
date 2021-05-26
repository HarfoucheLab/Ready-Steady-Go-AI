import torch
import numpy as np
from torch import nn
from torch import optim
import os
from torchvision import datasets, transforms, models
from collections import OrderedDict

from ..dcnnModelBuilder import *

def test_make_model():
    LEARNING_RATE = 0.0001
    ARCH = 'densenet161'
    hidden_layers = [10240, 1024]
    model = make_model(ARCH, hidden_layers, LEARNING_RATE, True)
    assert model.classifier.fc1.in_features == 2208
    
def test_loadModelWeights():
    DENSENET_PRETRAINED_PATH ='/content/models/RSGAI_DenseNet.pth'
    model = loadModelWeights(DENSENET_PRETRAINED_PATH)
    assert model.classifier.fc2.in_features == 10240
