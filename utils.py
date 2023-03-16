import os
import torch 

import numpy as np
import pandas as pd
from PIL import Image

from sklearn.model_selection import train_test_split
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms


def weight_vec(network):
    A = []
    for w in network.parameters():
        A.append(torch.flatten(w))
    return torch.cat(A)


def weight_dec_global(pyModel, weight_vec): 
    """
    Reshape the weight back to its original shape in pytorch and then 
    plug it to the model
    """
    c = 0
    for w in pyModel.parameters():
        m = w.numel()
        D = weight_vec[c:m+c].reshape(w.data.shape) 
        c+=m
        if w.data is None:
            w.data = D+0
        else:
            with torch.no_grad():
                w.set_( D+0 )
    return pyModel


def distribute_data(numOfClients, train_dataset, batch_size):
    """
    numOfClients: int 
    train_dataset: train_dataset (torchvision.datasets class)
    return distributed dataloaders for each client
    """
    # distribution list to fill the number of samples in each entry for each client
    distribution = []
    # rounding the number to get the number of dataset each client will get
    p = round(1/numOfClients * len(train_dataset))
    
    # the remainder data that won't be able to split if it's not an even number
    remainder_data = len(train_dataset) - numOfClients * p 
    # if the remainder data is 0 ---> all clients will get the same number of dataset
    if remainder_data == 0: 
        distribution = [p for i in range(numOfClients)]
    else:
        distribution = [p for i in range(numOfClients-1)]
        distribution.append(p+remainder_data)

    # splitting the data to different dataloaders
    data_split = torch.utils.data.random_split(train_dataset, distribution)
    # CLIENTS DATALOADERS
    ClIENTS_DATALOADERS = [torch.utils.data.DataLoader(data_split[i], batch_size=batch_size,shuffle=True, num_workers=32) for i in range(numOfClients)]
    
    print(f"Length of the training dataset: {len(train_dataset)} sample")
    return ClIENTS_DATALOADERS

def get_data(base_dir, imageid_path_dict):

    """
        Preprocessing for the SkinCancer dataset. 
        Input: 
            base_dir (str): path of the directory includes SkinCancer images
            imageid_path_dict (dict): dictionary with image id as keys and image pth as values
        
        Return: 
            df_train: Dataframe for training 
            df_val: Dataframe for testing 

    """

    lesion_type_dict = {
    'nv': 'Melanocytic nevi',
    'mel': 'dermatofibroma',
    'bkl': 'Benign keratosis-like lesions ',
    'bcc': 'Basal cell carcinoma',
    'akiec': 'Actinic keratoses',
    'vasc': 'Vascular lesions',
    'df': 'Dermatofibroma'
    }

    df_original = pd.read_csv(os.path.join(base_dir, 'HAM10000_metadata.csv'))
    df_original['path'] = df_original['image_id'].map(imageid_path_dict.get)
    df_original['cell_type'] = df_original['dx'].map(lesion_type_dict.get)
    df_original['cell_type_idx'] = pd.Categorical(df_original['cell_type']).codes

    df_original[['cell_type_idx', 'cell_type']].sort_values('cell_type_idx').drop_duplicates()

    # Get number of images associated with each lesion_id
    df_undup = df_original.groupby('lesion_id').count()
    # Filter out lesion_id's that have only one image associated with it
    df_undup = df_undup[df_undup['image_id'] == 1]
    df_undup.reset_index(inplace=True)

    # Identify lesion_id's that have duplicate images and those that have only one image.
    def get_duplicates(x):
        unique_list = list(df_undup['lesion_id'])
        if x in unique_list:
            return 'unduplicated'
        else:
            return 'duplicated'

    # create a new colum that is a copy of the lesion_id column
    df_original['duplicates'] = df_original['lesion_id']

    # apply the function to this new column
    df_original['duplicates'] = df_original['duplicates'].apply(get_duplicates)

    # Filter out images that don't have duplicates
    df_undup = df_original[df_original['duplicates'] == 'unduplicated']

    # Create a val set using df because we are sure that none of these images have augmented duplicates in the train set
    y = df_undup['cell_type_idx']
    _, df_val = train_test_split(df_undup, test_size=0.2, random_state=101, stratify=y)


    # This set will be df_original excluding all rows that are in the val set
    # This function identifies if an image is part of the train or val set.
    def get_val_rows(x):
        # create a list of all the lesion_id's in the val set
        val_list = list(df_val['image_id'])
        if str(x) in val_list:
            return 'val'
        else:
            return 'train'

    # Identify train and val rows
    # Create a new colum that is a copy of the image_id column
    df_original['train_or_val'] = df_original['image_id']
    # Apply the function to this new column
    df_original['train_or_val'] = df_original['train_or_val'].apply(get_val_rows)
    # Filter out train rows
    df_train = df_original[df_original['train_or_val'] == 'train']

    # Copy fewer class to balance the number of 7 classes
    data_aug_rate = [15,10,5,50,0,40,5]
    for i in range(7):
        if data_aug_rate[i]:
            df_train=df_train.append([df_train.loc[df_train['cell_type_idx'] == i,:]]*(data_aug_rate[i]-1), ignore_index=True)
    df_train['cell_type'].value_counts()

    df_train = df_train.reset_index()
    df_val = df_val.reset_index()

    return df_train, df_val

class CustomDataset(Dataset):
    """
        Cutom dataset for SkinCancer dataset
    """
    def __init__(self, df, transform=None):
        self.df = df
        self.transform = transform

    def __len__(self):
        return len(self.df)

    def __getitem__(self, index):
        # Load data and get label
        X = Image.open(self.df['path'][index])
        y = torch.tensor(int(self.df['cell_type_idx'][index]))

        if self.transform:
            X = self.transform(X)
        return X, y

class ISIC2019(Dataset): 


    TO_REPLACE_TRAIN = [None, [4,5,6], None, None,[4], [4,5,6]]
    VALUES_TRAIN = [None, [3,4,5], None, None,[2], [3,4,5]]

    def __init__(self, csv_file_path, root_dir, client_id, train = True, centralized = False, input_size = 224) -> None:
        super().__init__()
        self.image_root = root_dir   
        self.train = train  
        csv_file = pd.read_csv(csv_file_path)
        self.centralized = centralized

        if train:
            if centralized: 
                self.csv = csv_file[csv_file['fold'] == 'train'].reset_index()
            else:
                self.csv = csv_file[csv_file['fold2'] == f'train_{client_id}'].reset_index()

        elif train == False:  
            if centralized: 
                self.csv = csv_file[csv_file['fold'] == 'test'].reset_index()
            else: 
                self.csv = csv_file[csv_file['fold2'] == f'test_{client_id}'].reset_index()

        if train:
            self.transform = transforms.Compose([
                transforms.RandomRotation(10), 
                transforms.RandomHorizontalFlip(0.5), 
                transforms.RandomVerticalFlip(0.5), 
                transforms.RandomAffine(degrees = 0, shear=0.05),
                transforms.RandomResizedCrop((input_size, input_size), scale=(0.85,1.1)),
                transforms.ToTensor(), 
            ])
            
        elif train == False:
           self.transform = transforms.Compose([
                    transforms.Resize((input_size, input_size)),
                    transforms.ToTensor(),
                ])
    def __len__(self):
        return self.csv.shape[0]

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        img_name = os.path.join(self.image_root,
                                self.csv['image'][idx]+'.jpg')
        sample = Image.open(img_name)
        target = self.csv['target'][idx]

        sample = self.transform(sample)

        return sample, target

def blood_noniid(numOfAgents, data, batch_size):
    """
        Function to divide the bloodmnist among clients 

        Input: 
            numOfAgents (int): Number of Agents (Clients)
            data: dataset to be divided 
            batch_size (int)

    
        Return: 
            datasets for agents, Loaders for agents , datasets for visualization

    """
    # static way of creating non iid data, to change the distribution change the index of p in
    # the for loop 
    nonIID_tensors = [[] for i in range(numOfAgents)]  
    nonIID_labels = [[] for i in range(numOfAgents)]  
    agents = np.arange(0,numOfAgents)
    c = 0
    p = np.ones((numOfAgents))
    xx = 0
    for i in data:
        xx+=1
        p = np.ones((numOfAgents))
        if float(i[1]) == 0:
            p[0] = numOfAgents
            p[1] = numOfAgents
            p[2] = numOfAgents
        if float(i[1]) == 1:
            p[0] = numOfAgents
            p[1] = numOfAgents
            p[2] = numOfAgents
        if float(i[1]) == 2:
            p[3] = numOfAgents
            p[5] = numOfAgents
            p[0] = numOfAgents
        if float(i[1]) == 3:
            p[0] = numOfAgents
            p[4] = numOfAgents
            p[5] = numOfAgents
        if float(i[1]) == 4:
            p[3] = numOfAgents
            p[4] = numOfAgents
            p[5] = numOfAgents
        if float(i[1]) == 5:
            p[3] = numOfAgents
            p[4] = numOfAgents
            p[5] = numOfAgents
        if float(i[1]) == 6:
            p[4] = numOfAgents
            p[5] = numOfAgents
            p[5] = numOfAgents
        if float(i[1]) == 7:
            p[0] = numOfAgents
            p[1] = numOfAgents
            p[2] = numOfAgents
        p = p / np.sum(p)
        j = np.random.choice(agents, p = p)
        nonIID_tensors[j].append(i[0])
        nonIID_labels[j].append(torch.tensor(i[1]).reshape(1))
    
    dataset_vis = [[] for i in range(numOfAgents) ]
    for i in range(numOfAgents):
        dataset_vis[i].append((torch.stack(nonIID_tensors[i]),torch.cat(nonIID_labels[i])))
    
    dataset_agents = [[] for i in range(numOfAgents) ]
    for agent in range(numOfAgents): 
        im_ = dataset_vis[agent][0][0]
        lab_ =  dataset_vis[agent][0][1]
        for im, lab in zip(im_, lab_):
            dataset_agents[agent].append((im, lab))

    dataset_loaders = [DataLoader(dataset_agents[i], batch_size=batch_size, shuffle=True, num_workers=8) for i in range(numOfAgents)]
    
    return dataset_agents, dataset_loaders, dataset_vis