import sys
import os
import glob

import numpy as np 

from PIL import Image
from torch.utils.data import DataLoader
from torchvision import transforms
import torch.utils.data as data
import torch 

import medmnist
from medmnist import INFO

from utils import get_data, CustomDataset, ISIC2019, blood_noniid, distribute_data

import random 

seed = 105
np.random.seed(seed)
torch.manual_seed(seed)
random.seed(seed)


def distribute_images(dataset_name,train_data, num_clients, test_data, batch_size, num_workers = 8):
    """
    This method splits the dataset among clients.
    train_data: train dataset 
    test_data: test dataset 
    batch_size: batch size

    """
    if dataset_name == 'HAM':
        CLIENTS_DATALOADERS = distribute_data(num_clients, train_data, batch_size)
        testloader = torch.utils.data.DataLoader(test_data,batch_size=batch_size, num_workers= num_workers)

    elif dataset_name == 'bloodmnist':
        _, testloader, train_dataset, _ = bloodmnisit(batch_size= batch_size)
        _, CLIENTS_DATALOADERS, _ = blood_noniid(num_clients, train_dataset, batch_size =batch_size)
        
    return CLIENTS_DATALOADERS, testloader

def bloodmnisit(input_size =224, batch_size = 32, num_workers= 8, download = True):
    """
        Get train/test loaders and sets for bloodmnist from medmnist library. 

        Input: 
            input_size (int): width of the input image which issimilar to height 
            batch_size (int)
            num_workers (int): Num of workeres used for in creating the loaders 
            download (bool): Whether to download the dataset or not
        
        return: 
            train_loader, test_loader, train_dataset, test_dataset
    """

    data_flag = 'bloodmnist'
    info = INFO[data_flag]
    DataClass = getattr(medmnist, info['python_class'])

    data_transform_train = transforms.Compose([
        transforms.RandomVerticalFlip(),
        transforms.RandomHorizontalFlip(),
        transforms.RandomAffine(degrees= 10, translate=(0.1,0.1)),
        transforms.RandomResizedCrop(input_size, (0.75,1), (0.9,1)), 
        transforms.ToTensor(),
        ]) 
    
    data_transform_teest = transforms.Compose([
        transforms.Resize(224), 
        transforms.ToTensor(),
        ])
    
    train_dataset = DataClass(split='train', transform=data_transform_train, download=download)
    test_dataset = DataClass(split='test', transform=data_transform_teest, download=download)

    train_loader = data.DataLoader(dataset=train_dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers)
    test_loader = data.DataLoader(dataset=test_dataset, batch_size=2*batch_size, shuffle=False, num_workers=num_workers)
    
    return train_loader, test_loader, train_dataset, test_dataset

def skinCancer(input_size = 224, batch_size = 32, base_dir = './data', num_workers = 8):
    """
        Get the SkinCancer datasets and dataloaders. 

        Input:
            input_size (int): width of the input image
            batch_size (int)
            base_dir (str): Path to directory which includes the skincancer images
            num_workers (int): for dataloaders 
        
        return: 
            train_loader, testing_loader, train_dataset, test_dataset
    
    """
    all_image_path = glob.glob(os.path.join(base_dir, '*.jpg'))
    imageid_path_dict = {os.path.splitext(os.path.basename(x))[0]: x for x in all_image_path}
    df_train, df_val = get_data(base_dir, imageid_path_dict)

    normMean = [0.76303697, 0.54564005, 0.57004493]
    normStd = [0.14092775, 0.15261292, 0.16997]

    train_transform = transforms.Compose([transforms.RandomResizedCrop((input_size,input_size), scale=(0.9,1.1)),
                                          transforms.ColorJitter(brightness=0.1, contrast=0.1, hue=0.1),
                                          transforms.RandomRotation(10),
                                          transforms.RandomHorizontalFlip(),
                                          transforms.ToTensor(),
                                          transforms.Normalize(normMean, normStd)])

    # define the transformation of the val images.
    val_transform = transforms.Compose([transforms.Resize((input_size,input_size)),
                                        transforms.ToTensor(),
                                        transforms.Normalize(normMean, normStd)])

    training_set = CustomDataset(df_train.drop_duplicates('image_id'), transform=train_transform)
    train_loader = DataLoader(training_set, batch_size=batch_size, shuffle=True, num_workers=num_workers)

    # Same for the validation set:
    validation_set = CustomDataset(df_val.drop_duplicates('image_id'), transform=val_transform)
    val_loader = DataLoader(validation_set, batch_size=batch_size, shuffle=False, num_workers=num_workers)
    
    return train_loader, val_loader, training_set, validation_set

def isic2019(input_size = 224, root_dir = './ISIC_2019_Training_Input_preprocessed', csv_file_path = './train_test_split', batch_size = 32, num_workers=8):
    
    """
        Function that return train and test dataloaders and datasets fir centralized training and federated settings. 

        Input: 
            root_dir (str): path to directory that has preproceessed images from FLamby library
            csv_file_path (str): Path to the csv file that has train_test_split as per FLamby Library
        
        Return: 
            Clients train dataloaders (federated), Clients test loaders, Train dataloader (centralized), 
            Clients train datasets (Federated), Clients test datasets (Federated), Test dataloader (All testing images in one loader) 
    """
    clients_datasets_train = [
        ISIC2019(
        csv_file_path= csv_file_path, 
        root_dir=root_dir,client_id=i,train=True, centralized=False, input_size= input_size) for i in range(6)
    ]
    
    test_datasets = [
         ISIC2019(
        csv_file_path= csv_file_path, 
        root_dir=root_dir, client_id=i, train=False, centralized=False, input_size= input_size) for i in range(6)
        
    ]
    
    centralized_dataset_train = ISIC2019(
        csv_file_path= csv_file_path, 
        root_dir=root_dir, client_id=None ,train=True, centralized=True, input_size= input_size
    )
    
    clients_dataloader_train = [
        DataLoader(
        dataset=clients_datasets_train[i],batch_size= batch_size, shuffle=True, num_workers=num_workers
        ) for i in range(6)
    ]
    
    test_dataloaders = [
        DataLoader(dataset=test_datasets[i],batch_size= batch_size, shuffle=False, num_workers=num_workers)
        for i in range(6)
    ]

    test_centralized_dataset =  ISIC2019(
        csv_file_path= csv_file_path, 
        root_dir=root_dir, client_id=None , train=False, centralized=True, input_size= input_size
        )

    test_dataloader_centralized = DataLoader(dataset=test_centralized_dataset,batch_size= batch_size, shuffle=False, num_workers=num_workers)

    
    centralized_dataloader_train = DataLoader(dataset=centralized_dataset_train,batch_size= batch_size, shuffle=True, num_workers=num_workers)

    return clients_dataloader_train, test_dataloaders, centralized_dataloader_train, clients_datasets_train, test_datasets, test_dataloader_centralized