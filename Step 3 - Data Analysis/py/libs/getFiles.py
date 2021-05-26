import requests
import os
import zipfile

def DownloadPlantVillageCroppedSegmentedDataset(dataset_url, save_data_to, dataset_file_name):
  
    if not os.path.exists(save_data_to):
        os.makedirs(save_data_to)

    r = requests.get(dataset_url, stream = True, headers={"User-Agent": "Ready, Steady, Go AI"})

    print("Downloading dataset...")  

    with open(save_data_to + dataset_file_name, "wb") as file: 
        for block in r.iter_content(chunk_size = 1024):
            if block: 
                file.write(block)

    ## Extract downloaded zip dataset file
    print("Dataset downloaded")  
    print("Extracting files...")  
    with zipfile.ZipFile(save_data_to + dataset_file_name, 'r') as zip_dataset:
        zip_dataset.extractall(save_data_to)

    ## Delete the zip file as we no longer need it
    os.remove(save_data_to + dataset_file_name)
    print("All done!")  


def DownloadPretrainedDCNNDenseNet161TomatoModel(model_URL, save_data_to, model_file_name):

    if not os.path.exists(save_data_to):
        os.makedirs(save_data_to)

    print("Downloading model...")  

    r = requests.get(model_URL, stream = True, headers={"User-Agent": "Ready, Steady, Go AI"})
    with open(save_data_to + model_file_name, "wb") as file: 
        for block in r.iter_content(chunk_size = 1024):
            if block: 
                file.write(block)

    ## Extract downloaded zip dataset file
    print("Model downloaded")  
    print("Extracting files...")

    with zipfile.ZipFile(save_data_to + model_file_name, 'r') as zip_dataset:
        zip_dataset.extractall(save_data_to)
    print("All done!")  