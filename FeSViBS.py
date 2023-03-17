import os 
import numpy as np
import models 
import random
from dataset import skinCancer, bloodmnisit, isic2019
from utils import weight_dec_global, weight_vec
import argparse 
import torch as torch
from torch import nn




def fesvibs(
        dataset_name, lr, batch_size, Epochs, input_size, num_workers, save_every_epochs, 
        model_name, pretrained, opt_name, seed, base_dir, root_dir, csv_file_path, num_clients, DP, 
        epsilon, delta, resnet_dropout, initial_block, final_block, fesvibs_arg, local_round
        ):

    torch.manual_seed(seed)
    random.seed(seed)
    np.random.seed(seed)

    if fesvibs_arg: 
        method_flag = 'FeSViBS'
    else:
        method_flag = 'SViBS'

    if torch.cuda.is_available():
        device = 'cuda'
    else: 
        device = 'cpu'

    if DP:
        std = np.sqrt(2 * np.math.log(1.25/delta)) / epsilon 
        mean=0
        dir_name = f"{model_name}_{lr}lr_{dataset_name}_{num_clients}Clients_{initial_block}to{final_block}Blocks_{batch_size}Batch__{epsilon,delta}DP_{method_flag}"
    else:
        mean = 0
        std = 0
        dir_name = f"{model_name}_{lr}lr_{dataset_name}_{num_clients}Clients_{initial_block}to{final_block}Blocks_{batch_size}Batch_{method_flag}"

    save_dir = f'{dir_name}' 
    os.mkdir(save_dir)    

    print(f"Logging to: {dir_name}")

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

    criterion = nn.CrossEntropyLoss()

    fesvibs_network = models.FeSVBiS(
            ViT_name= model_name, num_classes= num_classes,
            num_clients = num_clients, in_channels = num_channels,
            ViT_pretrained= pretrained,
            initial_block= initial_block, final_block= final_block,
            resnet_dropout= resnet_dropout, DP=DP, mean= mean, std= std
            ).to(device)
    
    Split = models.SplitFeSViBS(
        num_clients=num_clients, device = device, network = fesvibs_network, 
        criterion = criterion, base_dir=save_dir, 
        initial_block= initial_block, final_block= final_block,
        )
    

    if dataset_name != 'isic2019':
        print('Distribute Images Among Clients')
        Split.distribute_images(dataset_name=dataset_name, train_data= traindataset,test_data= testdataset ,batch_size = batch_size)  
    else: 
        Split.CLIENTS_DATALOADERS = DATALOADERS
        Split.testloader = test_loader

    Split.set_optimizer(opt_name, lr = lr)
    Split.init_logs()

    print('Start Training! \n')

    for r in range(Epochs):
        print(f"Round {r+1} / {Epochs}")
        agg_weights = None
        for client_i in range(num_clients):
            weight_dict = Split.train_round(client_i)
            if client_i == 0: 
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

        if fesvibs_arg and ((r+1) % local_round == 0 and r!= 0):
                print('========================== \t \t Federation \t \t ==========================')
                tails_weights = []
                head_weights = []
                for head, tail in zip(Split.network.resnet50_clients, Split.network.mlp_clients_tail):
                    head_weights.append(weight_vec(head).detach().cpu())
                    tails_weights.append(weight_vec(tail).detach().cpu())
                
                mean_avg_tail = torch.mean(torch.stack(tails_weights), axis = 0)
                mean_avg_head = torch.mean(torch.stack(head_weights), axis = 0)

                for i in range(num_clients):
                    Split.network.mlp_clients_tail[i] = weight_dec_global(Split.network.mlp_clients_tail[i], 
                                                                        mean_avg_tail.to(device))
                    Split.network.resnet50_clients[i] = weight_dec_global(Split.network.resnet50_clients[i], 
                                                                        mean_avg_head.to(device))
       
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
    parser.add_argument('--local_round',  type=int, default= 2, help='Local round before federation in FeSViBS, default : 2')
    parser.add_argument('--num_workers',  type=int, default= 8, help='Number of workers for dataloaders, default : 8')
    parser.add_argument('--initial_block',  type=int, default= 1, help='Initial Block, default : 1')
    parser.add_argument('--final_block',  type=int, default= 6, help='Final Block, default : 6')
    parser.add_argument('--num_clients',  type=int, default= 6, help='Number of Clients, default : 6')
    parser.add_argument('--model_name', type=str, default= 'vit_base_r50_s16_224', help='Model name from timm library, default: vit_base_r50_s16_224')
    parser.add_argument('--pretrained', type=bool, default= False, help='Pretrained weights flag, default: False')
    parser.add_argument('--fesvibs_arg', type=bool, default= False, help='Flag to indicate whether SViBS or FeSViBS, default: False')
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
    parser.add_argument('--resnet_dropout',  type=float, default= 0.5, help='ResNet Dropout, Default: 0.5')
    args = parser.parse_args()

    fesvibs(
        dataset_name = args.dataset_name, input_size= args.input_size, 
        num_workers= args.num_workers, model_name= args.model_name, 
        pretrained= args.pretrained, batch_size= args.batch_size, 
        Epochs= args.Epochs, opt_name= args.opt_name, lr= args.lr, 
        save_every_epochs= args.save_every_epochs, seed= args.seed, 
        base_dir= args.base_dir, root_dir= args.root_dir, csv_file_path= args.csv_file_path,  num_clients = args.num_clients, 
        DP = args.DP, epsilon = args.epsilon, delta = args.delta, initial_block= args.initial_block, final_block=args.final_block,
        resnet_dropout = args.resnet_dropout, fesvibs_arg = args.fesvibs_arg, local_round = args.local_round
        )