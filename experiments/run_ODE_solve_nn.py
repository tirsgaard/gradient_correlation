from typing import Iterable
import torch
import torchvision
import scipy
from src.models.NN_models import SingleLayerMLP, SimpleMLP
import yaml
from datasets.download_MNIST import get_MNIST
from src.training import train_epoch, validate, test
from src.correlation_gradient import get_gradient, rank_sample_information, rank_correlation_uniqueness, construct_covariance_matrix
from tqdm import tqdm
from easydict import EasyDict as edict
import matplotlib.pyplot as plt
from copy import deepcopy
from torch.utils.data.dataloader import default_collate

import numpy as np

def svd_pseudo_inverse(matrix: torch.Tensor, k: int) -> torch.Tensor:
    u, s, v = torch.svd(matrix)
    s_inv = 1/s
    s_inv[k:] = 0
    return v @ torch.diag(s_inv) @ u.T

class GaussianFit(torch.nn.Module):
    def __init__(self, model: torch.nn.Module):
        super(GaussianFit, self).__init__()
        self.model = model
        
    def fit(self, data: Iterable[torch.Tensor]):
        xs, ys, y_hats = [], [], []
        with torch.no_grad():
            for x, y in data:
                xs.append(x)
                ys.append(y)
                y_hats.append(torch.softmax(self.model(x.to(device)), -1))
        xs = torch.cat(xs, 0).to(device)
        y = torch.cat(ys, 0).to(device)
        y_hat = torch.cat(y_hats, 0).to(device)
        label_diff = y - y_hat
        self.grads = get_gradient(self.model, xs, loss_batched, optimizer, True, True, y=y, pKernel=True)
        #self.grads = self.grads - self.grads.mean(-1, keepdim=True)
        covarinace_matrix = self.grads@self.grads.T
        covarinace_matrix = covarinace_matrix
        self.W = torch.linalg.solve(covarinace_matrix.cpu(), label_diff.cpu()).to(device)
        #self.W = svd_pseudo_inverse(covarinace_matrix, 10) @ label_diff
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x_grad = get_gradient(self.model, x, loss_batched, optimizer, True, True, pKernel=True)
        #x_grad = x_grad - x_grad.mean(-1, keepdim=True)
        K_xX = x_grad@self.grads.T
        with torch.no_grad():
            y_hat = torch.softmax(self.model(x), -1)
        return y_hat + K_xX @ self.W
    
def MSELoss(y_hat, y):
    return 0.5*(y_hat-y).pow(2).sum(-1).mean()

def MSELoss_batch(y_hat, y):
    return 0.5*(y_hat-y).pow(2).sum(-1)

config = edict(yaml.safe_load(open('configs/MNIST.yaml', 'r')))
device = torch.device('cpu')  #torch.device('cuda' if torch.cuda.is_available() else ("mps" if torch.backends.mps.is_available() else 'cpu'))
def move_collate(batch):
    return tuple(x.to(device) for x in default_collate(batch))

if __name__ == '__main__':
    torch.manual_seed(1)
    loss = MSELoss
    loss_batched = MSELoss_batch
    epochs = 10
    # Load the data
    train_data, val_data, test_data = get_MNIST(0.1)
    
    gradient_batch_size = 128
    # Shuffle the train data
    train_data = torch.utils.data.Subset(train_data, torch.randperm(len(train_data)))
    val_data = torch.utils.data.Subset(val_data, torch.randperm(len(val_data)))

    collate_func = move_collate
    small_training_set = torch.utils.data.Subset(train_data, range(gradient_batch_size))
    small_validation_set = torch.utils.data.Subset(val_data, range(gradient_batch_size))
    train_loader = torch.utils.data.DataLoader(small_training_set, batch_size=gradient_batch_size, shuffle=True)
    val_loader = torch.utils.data.DataLoader(small_validation_set, batch_size=gradient_batch_size, shuffle=False, pin_memory=True)
    gradient_loader = torch.utils.data.DataLoader(small_training_set, batch_size=gradient_batch_size, shuffle=False, pin_memory=True)

    validation_acc = torch.zeros((3, epochs))

    # First 100 hidden units
    N_hidden = 256
    model = SingleLayerMLP(input_size=784, output_size=10, hidden_size=N_hidden).to(device)
    initial_model = deepcopy(model)
    optimizer = torch.optim.SGD(model.parameters(), lr=config.training.learning_rate, momentum=0.9)
    epoch = 20

    def validate_model(model, val_loader):
        model.eval()
        y_hat = torch.cat([model(x.to(device)) for x, _ in val_loader], 0)
        y_val = torch.cat([y for _, y in val_loader], 0).to(device)
        acc = (y_hat.argmax(-1) == y_val.argmax(-1)).float().mean(-1)
        return acc


    # Train the model and fit the kernel
    validation_acc_list = []
    validation_kernel_acc_list = []
    for j in tqdm(range(epochs)): 
        kernel_model = GaussianFit(model)
        kernel_model.fit(gradient_loader)
        validation_kernel_acc_list.append(validate_model(kernel_model, val_loader))
        validation_acc_list.append(validate_model(model, val_loader))
        model.train()
        for j in range(500):
            train_epoch(model, optimizer, loss, train_loader, device)
    validation_acc = torch.stack(validation_acc_list).cpu()
    validation_kernel_acc = torch.stack(validation_kernel_acc_list).cpu()


    small_training_set = torch.utils.data.Subset(train_data, range(gradient_batch_size))
    small_validation_set = torch.utils.data.Subset(val_data, range(gradient_batch_size))
    train_loader = torch.utils.data.DataLoader(small_training_set, batch_size=config.training.batch_size, shuffle=True, pin_memory=True)
    val_loader = torch.utils.data.DataLoader(small_validation_set, batch_size=1, shuffle=False, pin_memory=True)
    gradient_loader = torch.utils.data.DataLoader(small_training_set, batch_size=1, shuffle=False, pin_memory=True)


    # First 100 hidden units
    model = initial_model
    optimizer = torch.optim.SGD(model.parameters(), lr=config.training.learning_rate, momentum=0.)

    # Train the model and fit the kernel
    validation_acc_list = []
    validation_kernel_acc_list = []
    for j in tqdm(range(epochs)):
        kernel_model = GaussianFit(model)
        kernel_model.fit(gradient_loader)
        validation_kernel_acc_list.append(validate_model(kernel_model, val_loader))
        validation_acc_list.append(validate_model(model, val_loader))
        model.train()
        for j in range(50):
            train_epoch(model, optimizer, loss, train_loader, device)
            
    validation_acc_full = torch.stack(validation_acc_list).cpu()
    validation_kernel_acc_full = torch.stack(validation_kernel_acc_list).cpu()

    plt.figure()
    plt.plot(validation_acc, label='Traning model GD', color='blue')
    plt.plot(validation_kernel_acc, label='Kernel model GD', color='blue', linestyle='--')
    #plt.plot(validation_acc_full, label='Traning model SGD', color='red')
    #plt.plot(validation_kernel_acc_full, label='Kernel model SGD', color='red', linestyle='--')

    plt.xlabel('Epoch*10')
    plt.ylabel('Accuracy')
    plt.legend()
    plt.savefig('kernel_fit.svg')
    plt.show()

    """    
    t_vals = torch.linspace(0, 1, 100)
    y_vals = torch.zeros((len(t_vals), len(y_hat), 10))

    with torch.no_grad():
        for i in range(len(t_vals)):
            res = right_side_function(x_val, t_vals[i])
            y_vals = y_hat + K_xX @ res

    # Plot accuracy as a function of t
    plt.figure()
    acc = (y_vals.argmax(-1) == y_val).float().mean(-1)
    plt.plot(t_vals, acc)
    plt.xlabel('t')
    plt.ylabel('Accuracy')
    plt.legend()
    plt.show(),
    """