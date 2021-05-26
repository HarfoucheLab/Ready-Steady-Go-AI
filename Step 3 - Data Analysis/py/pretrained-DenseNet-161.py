import libs

## FEEL FREE TO CHANGE THESE PARAMETERS
tomato_dataset_url = "http://faridnakhle.com/pv/tomato-split-cropped-segmented-balanced.zip"
save_tomato_dataset_to = "/content/dataset/tomato-dataset/"
save_tomato_dataset_as = "tomato-split-cropped-segmented-balanced.zip"

    
tomato_densenet161_model_url = "http://faridnakhle.com/pv/models/RSGAI_DenseNet.zip"
save_densenet_model_to = "/content/models/"
save_densenet_model_as = "densenet.zip"
DENSENET_PRETRAINED_PATH ='/content/models/RSGAI_DenseNet.pth'
#######################################

'''
We start by downloading the dataset previously cropped and segmented using
the YOLO and SEGNET algorithms, and balanced by KNN, Augmentor, and DCGAN
'''
libs.DownloadPlantVillageCroppedSegmentedDataset(tomato_dataset_url, save_tomato_dataset_to, save_tomato_dataset_as)

'''
Next, Let us import some modules required for PyTorch
which we will be using to download and test our model
'''
import os
import matplotlib.pyplot as plt
import torch
import numpy as np
from torch import nn
from torch import optim
import torch.nn.functional as F
from torchvision import datasets, transforms, models
from PIL import Image
import json
''' ----------------------------------------- '''

'''
In the next block, we will define the global variables
for the DCNN algorithms. You can change these variables to your needs
'''
## YOU CAN CHANGE THESE VARIABLES    
EPOCHS = 100
BATCH_SIZE = 20
LEARNING_RATE = 0.0001
data_dir = '/content/dataset/tomato-dataset/'
save_checkpoints = True
save_model_to = '/content/output/'
if not os.path.exists(save_model_to):
    os.makedirs(save_model_to)
IMG_SIZE = 220
NUM_WORKERS = 1
print_every = 300
ARCH = 'densenet161'
hidden_layers = [10240, 1024]

device = 'cpu'
if torch.cuda.is_available():
      device = 'cuda'
  
######################################################

''' Preparing the datasets loaders: training, validation, and testing '''
train_loader, val_loader, test_loader, train_dataset, val_dataset, test_dataset = libs.data_loader(data_dir, IMG_SIZE, BATCH_SIZE, NUM_WORKERS, False)
print("Training Set: " + str(len(train_loader.dataset)))
print("Validation Set: " + str(len(val_loader.dataset)))
print("Testing Set: " + str(len(test_loader.dataset)))
''' -------------------------------------------------- '''

''' Now we can build our model '''
model = libs.make_model(ARCH, hidden_layers, LEARNING_RATE, True)
''' -------------------------- '''

# defining loss and optimizer
criterion = nn.NLLLoss()
optimizer = optim.Adam(model.classifier.parameters(), lr=LEARNING_RATE)

''' Download the pretrained DCNN-DenseNet161 model '''
libs.DownloadPretrainedDCNNDenseNet161TomatoModel(tomato_densenet161_model_url, save_densenet_model_to, save_densenet_model_as)

''' Now let's load the weights into our model '''
model = libs.loadModelWeights(DENSENET_PRETRAINED_PATH)
model.to(device)

''' We can finally test the model '''

with torch.no_grad():
    model.eval()
    test_preds, test_labels = libs.get_all_preds(model,test_loader)
    preds_correct = libs.get_num_correct(test_preds.to(device), test_labels.to(device))
    print('total correct:', preds_correct)
    print('accuracy:')
    print(((preds_correct / (len(test_loader.dataset))) * 100))

''' We now have the model accuraccy, but for better assessment, let us generate the confusion matrix '''
cmt = torch.zeros(10, 10, dtype=torch.int32) #10 is the number of classes
stacked = torch.stack(
    (
        test_labels
        ,test_preds.argmax(dim=1)
    )
    ,dim=1
)

for p in stacked:
    tl, pl = p.tolist()
    tl = int(tl)
    pl = int(pl)
    cmt[tl, pl] = cmt[tl, pl] + 1

#Plot CM
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix
cm = confusion_matrix(test_labels.cpu(), test_preds.argmax(dim=1).cpu())
plt.figure(figsize=(12, 12))
libs.plot_confusion_matrix(cm, test_dataset.classes, True, 'Confusion matrix', cmap=plt.cm.Blues)
plt.show()