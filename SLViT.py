import os
import torch 
import numpy as np
from torch import nn
import random 
from models import SLViT, SplitNetwork
from dataset import skinCancer, bloodmnisit, isic2019
import argparse 
from utils import weight_dec_global

def slvit(dataset_name, lr, batch_size, Epochs, input_size, num_workers, save_every_epochs, model_name, pretrained, opt_name, seed , base_dir, root_dir, csv_file_path, num_clients, DP, epsilon, delta):

    np.random.seed(seed)
    torch.manual_seed(seed)
    random.seed(seed)

    if torch.cuda.is_available():
        device = 'cuda'
    else:
        device = 'cpu'

    mean = 0 
    std  = 1
    if DP: 
        std = np.sqrt(2 * np.math.log(1.25/delta)) / epsilon 

    save_dir = f'{model_name}_{lr}lr_{dataset_name}_{num_clients}Clients_{DP}DP_{batch_size}Batch_SLViT'

    if DP: 
        save_dir = f'{model_name}_{lr}lr_{dataset_name}_{num_clients}Clients_({epsilon}, {delta})DP_{batch_size}Batch_SLViT'
    
    os.mkdir(save_dir)

    print('Getting the Dataset and Dataloader!')
    if dataset_name == 'HAM': 
        num_classes = 7
        _, _, traindataset, testdataset = skinCancer(input_size= input_size, batch_size = batch_size, base_dir= base_dir, num_workers=num_workers)
        num_channels = 3

    elif dataset_name == 'bloodmnist':
        num_classes = 8
        _, _, traindataset, testdataset = bloodmnisit(input_size= input_size, batch_size = batch_size, download= True, num_workers=num_workers)
        num_channels = 3

    elif dataset_name == 'isic2019': 
        num_classes = 8
        DATALOADERS, _, _, _, _, test_loader = isic2019(input_size= input_size, batch_size = batch_size, root_dir=root_dir, csv_file_path=csv_file_path, num_workers=num_workers)
        num_channels = 3

    slvit = SLViT(
        ViT_name= model_name, num_classes=num_classes,
        num_clients=num_clients, in_channels=num_channels,
        ViT_pretrained = pretrained,
        diff_privacy=DP, mean=mean, std = std
    ).to(device)

    criterion = nn.CrossEntropyLoss()

    Split = SplitNetwork(
        num_clients=num_clients, device = device, 
        network = slvit, criterion = criterion, base_dir=save_dir,
        )

    print('Distribute Data')
    if dataset_name != 'isic2019':  
        Split.distribute_images(dataset_name=dataset_name, train_data=traindataset, test_data=testdataset , batch_size = batch_size)  
    else:
        Split.CLIENTS_DATALOADERS  = DATALOADERS
        Split.testloader = test_loader

    Split.set_optimizer(opt_name, lr = lr)
    Split.init_logs()

    for r in range(Epochs):
        print(f"Round {r+1} / {Epochs}")
        agg_weights = None
        for client_i in range(num_clients):
            weight_dict = Split.train_round(client_i)
            if client_i ==0: 
                agg_weights = weight_dict
            else: 
                agg_weights['blocks'] +=  weight_dict['blocks']
                agg_weights['cls'] +=  weight_dict['cls']
                agg_weights['pos_embed'] +=  weight_dict['pos_embed']
        
        agg_weights['blocks'] /= num_clients
        agg_weights['cls'] /= num_clients
        agg_weights['pos_embed'] /= num_clients    

        Split.network.vit.blocks = weight_dec_global(
            Split.network.vit.blocks,
            agg_weights['blocks'].to(device)
            )
        
        Split.network.vit.cls_token.data = agg_weights['cls'].to(device) + 0.0
        Split.network.vit.pos_embed.data = agg_weights['pos_embed'].to(device) + 0.0
        
        for client_i in range(num_clients):
            Split.eval_round(client_i)
        
        print('---------')
        
        if (r+1) % save_every_epochs == 0 and r != 0: 
            Split.save_pickles(save_dir)

        print('============================================')

if __name__ == "__main__":

    parser = argparse.ArgumentParser(description='Run Centralized Experiments')

    parser.add_argument('--dataset_name', type=str, choices=['HAM', 'bloodmnist', 'isic2019'], help='Dataset Name')
    parser.add_argument('--input_size',  type=int, default= 224, help='Input size --> (input_size, input_size), default : 224')
    parser.add_argument('--num_workers',  type=int, default= 8, help='Number of workers for dataloaders, default : 8')
    parser.add_argument('--num_clients',  type=int, default= 6, help='Number of Clients, default : 6')
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
    parser.add_argument('--DP', type=bool, default= False, help='Differential Privacy , default: False')
    parser.add_argument('--epsilon',  type=float, default= 0, help='Epsilon Value for differential privacy')
    parser.add_argument('--delta',  type=float, default= 0.00001, help='Delta Value for differential privacy')


    args = parser.parse_args()

    slvit(
        dataset_name = args.dataset_name, input_size= args.input_size, 
        num_workers= args.num_workers, model_name= args.model_name, 
        pretrained= args.pretrained, batch_size= args.batch_size, 
        Epochs= args.Epochs, opt_name= args.opt_name, lr= args.lr, 
        save_every_epochs= args.save_every_epochs, seed= args.seed, 
        base_dir= args.base_dir, root_dir= args.root_dir, csv_file_path= args.csv_file_path,  num_clients = args.num_clients, 
        DP = args.DP, epsilon = args.epsilon, delta = args.delta
        )