import torch
import numpy as np
from torch import nn
from torch import optim
import os
from torchvision import datasets, transforms, models
from collections import OrderedDict

# Freeze parameters so we don't backprop through them
def make_model(structure, hidden_layers, lr, preTrained):
    if structure=="densenet161":
        model = models.densenet161(pretrained=preTrained)
        input_size = 2208
    else:
        model = models.vgg16(pretrained=preTrained)
        input_size = 25088
    output_size = 102
    for param in model.parameters():
        param.requires_grad = False

    classifier = nn.Sequential(OrderedDict([
                              ('dropout',nn.Dropout(0.5)),
                              ('fc1', nn.Linear(input_size, hidden_layers[0])),
                              ('relu1', nn.ReLU()),
                              ('fc2', nn.Linear(hidden_layers[0], hidden_layers[1])),
                              ('relu2', nn.ReLU()),
                              ('fc3', nn.Linear(hidden_layers[1], output_size)),
                              ('output', nn.LogSoftmax(dim=1))
                              ]))

    model.classifier = classifier
    return model

def loadModelWeights(modelPath):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    state = torch.load(modelPath, map_location=torch.device(device))
    lr = state['learning_rate']
    structure = state['structure']
    hidden_layers = state['hidden_layers']
    # Building the model from checkpoints
    model = make_model(structure, hidden_layers, lr, False)
    model.class_to_idx = state['class_to_idx']
    model.load_state_dict(state['state_dict'])
    model.eval()
    return model