import requests
import os
import zipfile

from ..getFiles import *

def test_DownloadPlantVillageCroppedSegmentedDataset():
    tomato_dataset_url = "http://faridnakhle.com/pv/tomato-split-cropped-segmented-balanced.zip"
    save_tomato_dataset_to = "/content/dataset/tomato-dataset/"
    save_tomato_dataset_as = "tomato-split-cropped-segmented-balanced.zip"
    DownloadPlantVillageCroppedSegmentedDataset(tomato_dataset_url, save_tomato_dataset_to, save_tomato_dataset_as)
    mainPathCreated = os.path.exists(save_tomato_dataset_to)
    train_dir = save_tomato_dataset_to + "train"
    val_dir = save_tomato_dataset_to + "val"
    test_dir = save_tomato_dataset_to + "test"
    trainPathExists = os.path.exists(train_dir)
    valPathExists = os.path.exists(val_dir)
    testPathExists = os.path.exists(test_dir)
    
    train_classes = [path for path in os.listdir(train_dir)]
    train_imgs = dict([(ID, os.listdir(os.path.join(train_dir, ID))) for ID in train_classes])
    train_classes_count = []
    train_file_count = 0
    for trainClass in train_classes:
        train_classes_count.append(len(train_imgs[trainClass]))
        train_file_count = train_file_count + len(train_imgs[trainClass])

    val_classes = [path for path in os.listdir(val_dir)]
    val_imgs = dict([(ID, os.listdir(os.path.join(val_dir, ID))) for ID in val_classes])
    val_classes_count = []
    val_file_count = 0
    for valClass in val_classes:
        val_classes_count.append(len(val_imgs[valClass]))
        val_file_count = val_file_count + len(val_imgs[valClass])

    test_classes = [path for path in os.listdir(test_dir)]
    test_imgs = dict([(ID, os.listdir(os.path.join(test_dir, ID))) for ID in test_classes])
    test_classes_count = []
    test_file_count = 0
    for testClass in test_classes:
        test_classes_count.append(len(test_imgs[testClass]))
        test_file_count = test_file_count + len(test_imgs[testClass])

    
    assert mainPathCreated == True
    assert trainPathExists == True
    assert valPathExists == True
    assert testPathExists == True
    assert train_file_count == 15228
    assert val_file_count == 1812
    assert test_file_count == 1825


def test_DownloadPretrainedDCNNDenseNet161TomatoModel():
    tomato_densenet161_model_url = "http://faridnakhle.com/pv/models/RSGAI_DenseNet.zip"
    save_densenet_model_to = "/content/models/"
    save_densenet_model_as = "densenet.zip"
    DENSENET_PRETRAINED_PATH ='/content/models/RSGAI_DenseNet.pth'
    DownloadPretrainedDCNNDenseNet161TomatoModel(tomato_densenet161_model_url, save_densenet_model_to, save_densenet_model_as)
    assert os.path.exists(save_densenet_model_to) == True
    assert os.path.isfile(DENSENET_PRETRAINED_PATH) == True