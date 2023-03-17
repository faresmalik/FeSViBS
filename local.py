import os 
import timm
import torch 
import numpy as np
from torch import nn
import os
import random 
import argparse 

from models import CentralizedFashion
from dataset import skinCancer, bloodmnisit, isic2019, distribute_images


def local(dataset_name, lr, batch_size, Epochs, input_size, num_workers, save_every_epochs, model_name, pretrained, opt_name, seed, base_dir, root_dir, csv_file_path, num_clients, local_arg):

    np.random.seed(seed)
    torch.manual_seed(seed)
    random.seed(seed)

    if torch.cuda.is_available():
        device = 'cuda'
    else:
        device = 'cpu'

    print('Load Dataset and DataLoader!')
    if dataset_name == 'HAM': 
        num_classes = 7
        train_loader, test_loader, train_data, test_data = skinCancer(input_size= input_size, batch_size = batch_size, base_dir= base_dir, num_workers=num_workers)
        num_channels = 3

    elif dataset_name == 'bloodmnist':
        num_classes = 8
        train_loader, test_loader, train_data, test_data = bloodmnisit(input_size= input_size, batch_size = batch_size, download= True, num_workers=num_workers)
        num_channels = 3

    elif dataset_name == 'isic2019': 
        num_classes = 8
        DATALOADERS, _, _, _, _, test_loader = isic2019(input_size= input_size, batch_size = batch_size, root_dir=root_dir, csv_file_path=csv_file_path, num_workers=num_workers)
        num_channels = 3
        


    print('Create Directory for metrics loggings!')
    save_dir = f'{model_name}_{lr}lr_{dataset_name}_{Epochs}rounds_Local'
    os.mkdir(save_dir)

    print(f'Train Local Fashion:\n Number of Clients :{num_clients}\n model: {model_name}\n dataset: {dataset_name}\n LR: {lr}\n Number of Epochs: {Epochs}\n Loggings: {save_dir}\n')

    if dataset_name in ['HAM', 'bloodmnist']:
        print(f'Distribute Dataset Among {num_clients} Clients')

        DATALOADERS, test_loader = distribute_images(
            dataset_name = dataset_name, train_data = train_data, num_clients= num_clients,
            test_data = test_data, batch_size = batch_size, num_workers= num_workers
            )

    print('Loading Model form timm Library for All clients!')
    model = [timm.create_model(
        model_name= model_name, 
        num_classes= num_classes, 
        in_chans = num_channels, 
        pretrained= pretrained,
    ).to(device) for i in range(num_clients)]
                
    criterion = nn.CrossEntropyLoss()

    local = [CentralizedFashion(
        device = device,
        network = model[i], criterion = criterion,
        base_dir = save_dir
    ) for i in range(num_clients)]


    for i in range(num_clients):
        local[i].set_optimizer(opt_name, lr = lr)
        local[i].init_logs()

    for r in range(Epochs):
        print(f"Round {r+1} / {Epochs}")
        for client_i in range(num_clients):
            print(f'Client {client_i+1} / {num_clients}')
            local[client_i].train_round(DATALOADERS[client_i])
            local[client_i].eval_round(test_loader)
            print('---------')
            if (r+1) % save_every_epochs == 0 and r != 0: 
                local[client_i].save_pickles(save_dir,local= local_arg, client_id=client_i+1) 
        print('============================================')


if __name__ == "__main__":

    parser = argparse.ArgumentParser(description='Run Centralized Experiments')

    parser.add_argument('--dataset_name', type=str, choices=['HAM', 'bloodmnist', 'isic2019'], help='Dataset Name')
    parser.add_argument('--num_clients',  type=int, default= 6, help='Number of clients, default : 6')
    parser.add_argument('--local_arg', type=bool, default= True, help='Local Argument, default: True')
    parser.add_argument('--input_size',  type=int, default= 224, help='Input size --> (input_size, input_size), default : 224')
    parser.add_argument('--num_workers',  type=int, default= 8, help='Number of workers for dataloaders, default : 8')
    parser.add_argument('--model_name', type=str, default= 'vit_base_r50_s16_224', help='Model name from timm library, default: vit_base_r50_s16_224')
    parser.add_argument('--pretrained', type=bool, default= False, help='Pretrained weights flag, default: False')
    parser.add_argument('--batch_size',  type=int, default= 32, help='Batch size, default : 32')
    parser.add_argument('--Epochs',  type=int, default= 200, help='Number of Epochs, default : 200')
    parser.add_argument('--opt_name', type=str, choices=['Adam'], default = 'Adam', help='Optimizer name, only ADAM optimizer is available')
    parser.add_argument('--lr',  type=float, default= 1e-4, help='Learning rate, default : 1e-4')
    parser.add_argument('--save_every_epochs',  type=int, default= 10, help='Save metrics every this number of epochs, default: 10')
    parser.add_argument('--seed',  type=int, default= 105, help='Seed, default: 105')
    parser.add_argument('--base_dir', type=str, default= None, help='')
    parser.add_argument('--root_dir', type=str, default= None, help='')
    parser.add_argument('--csv_file_path', type=str, default=None, help='')

    args = parser.parse_args()

    local(
        dataset_name = args.dataset_name, num_clients= args.num_clients, 
        input_size= args.input_size, local_arg= args.local_arg, 
        num_workers= args.num_workers, model_name= args.model_name, 
        pretrained= args.pretrained, batch_size= args.batch_size, 
        Epochs= args.Epochs, opt_name= args.opt_name, lr= args.lr, 
        save_every_epochs= args.save_every_epochs, seed= args.seed, 
        base_dir= args.base_dir, root_dir= args.root_dir, csv_file_path= args.csv_file_path
        )