from tqdm import tqdm
import pickle as pkl
import os 

import torch 
from sklearn.metrics import balanced_accuracy_score

class CentralizedFashion(): 
    def __init__(self, device, network, criterion, network_name, base_dir):
        """
            Class for Centralized Paradigm.    
            args:
                device: cuda vs cpu
                network: ViT model
                criterion: loss function to be used
                network_name: used for saving purposes 
                base_dir: where to save metrics as pickles
            return: 
                None 
        """
        self.device = device
        self.network = network
        self.criterion = criterion
        self.network_name = network_name
        self.base_dir = base_dir

    def set_optimizer(self, name, lr):
        """
        name: Optimizer name, e.g. Adam 
        lr: learning rate 

        """
        if name == 'Adam':
            self.optimizer = torch.optim.Adam(self.network.parameters(), lr = lr)

    def init_logs(self):
            """
            A method to initialize dictionaries for the metrics
            return : None 
            args: None
            """
            self.losses  = {'train':[], 'test':[]}
            self.balanced_accs = {'train':[], 'test':[]}

    def train_round(self, train_loader):
        """
        Training loop. 

        """
        running_loss = 0
        whole_labels = []
        whole_preds = []
        whole_probs = []
        for imgs, labels in tqdm(train_loader): 
            self.optimizer.zero_grad()
            imgs, labels = imgs.to(self.device),labels.to(self.device)
            output = self.network(imgs)
            labels = labels.reshape(labels.shape[0])
            loss = self.criterion(output, labels)
            loss.backward()
            self.optimizer.step()
            running_loss += loss.item() 
            _, predicted = torch.max(output, 1)
            whole_probs.append(torch.nn.Softmax(dim = -1)(output).detach().cpu())
            whole_labels.append(labels.detach().cpu())
            whole_preds.append(predicted.detach().cpu())    
        self.metrics(whole_labels, whole_preds, running_loss, len(train_loader), whole_probs, train = True)
        
    def eval_round(self, test_loader):
        """
        Evaluation loop. 

        client_i: Client index.
                
        """
        running_loss = 0
        whole_labels = []
        whole_preds = []
        whole_probs = []
        with torch.no_grad():
            for imgs, labels in tqdm(test_loader): 
                imgs, labels = imgs.to(self.device), labels.to(self.device)
                output = self.network(imgs)
                labels = labels.reshape(labels.shape[0])
                loss = self.criterion(output, labels)
                running_loss += loss.item() 
                _, predicted = torch.max(output, 1)
                whole_probs.append(torch.nn.Softmax(dim = -1)(output).detach().cpu())
                whole_labels.append(labels.detach().cpu())
                whole_preds.append(predicted.detach().cpu())    
            self.metrics(whole_labels, whole_preds, running_loss, len(test_loader), whole_probs, train= False)
    
    def metrics(self, whole_labels, whole_preds, running_loss, len_loader, whole_probs, train):
        """
        Save metrics as pickle files and the model as .pt file.
        
        """
        whole_labels = torch.cat(whole_labels)
        whole_preds = torch.cat(whole_preds)
        loss_epoch = running_loss/len_loader
        balanced_acc = balanced_accuracy_score(whole_labels.detach().cpu(),whole_preds.detach().cpu())
        if train == True:
            eval_name = 'train'
        else:
            eval_name = 'test'

        self.losses[eval_name].append(loss_epoch)
        self.balanced_accs[eval_name].append(balanced_acc)
        
        print(f"{eval_name}:")
        print(f"{eval_name}_loss :{loss_epoch:.3f}")
        print(f"{eval_name}_balanced_acc :{balanced_acc:.3f}")


    def save_pickles(self, base_dir, local= None, client_id=None): 
        if local and client_id: 
            with open(os.path.join(base_dir,f'loss_epoch_Client{client_id}'), 'wb') as handle:
                pkl.dump(self.losses, handle)
            with open(os.path.join(base_dir,f'balanced_accs{client_id}'), 'wb') as handle:
                pkl.dump(self.balanced_accs, handle)
        else: 
            with open(os.path.join(base_dir,'loss_epoch'), 'wb') as handle:
                pkl.dump(self.losses, handle)
            with open(os.path.join(base_dir,f'balanced_accs'), 'wb') as handle:
                pkl.dump(self.balanced_accs, handle)
