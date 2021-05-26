import torch
import numpy as np
from torch import nn
from torch import optim
import os
from torchvision import datasets, transforms, models
import itertools
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix

@torch.no_grad()
def get_all_preds(model, dataloader):
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        all_preds = torch.tensor([])
        all_preds = all_preds.to(device)
        all_labels = torch.tensor([])
        all_labels = all_labels.to(device)

        for data, target in dataloader:
            input = data.to(device)
            target = target.to(device)

            with torch.no_grad():
                output = model(input)

            all_preds = torch.cat(
                (all_preds, output)
                ,dim=0
            )
            all_labels = torch.cat(
                (all_labels, target)
                ,dim=0
            )

        return all_preds, all_labels
    
def get_num_correct(preds, labels):
        return preds.argmax(dim=1).eq(labels).sum().item()

def plot_confusion_matrix(cm, classes, normalize=False, title='Confusion matrix', cmap=plt.cm.Blues):
    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        #percentage: 
        cm = cm.astype('float') * 100
        # add percentage sign

    mycm = plt.imshow(cm, interpolation='nearest', cmap=cmap)
    mycm.set_clim([0,100])
    cbar = plt.colorbar(mycm, shrink=0.82, ticks=list(range(0, 120, 20)))
    cbar.ax.set_yticklabels(['0', '20', '40', '60', '80', '100'])  # vertically oriented colorbar

    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=45,  ha="right")
    
    plt.yticks(tick_marks, classes)

    fmt = '.2f' if normalize else 'd'
    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, str(format(cm[i, j], fmt)) + "%", horizontalalignment="center", color="white" if cm[i, j] > thresh else "black")
        

    plt.rcParams['font.family'] = "sans-serif"
    plt.rcParams['font.sans-serif'] = "Arial"
    plt.rcParams.update({'font.size': 12})
    plt.ylabel('True class', fontsize=17, fontweight='bold')
    plt.xlabel('Predicted class', fontsize=17, fontweight='bold')