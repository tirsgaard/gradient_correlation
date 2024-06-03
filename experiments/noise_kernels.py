import torch
import numpy as np
import matplotlib.pyplot as plt
from typing import Optional
from tqdm import tqdm
from src.models.NN_models import SimpleMLP, SingleLayerMLP
from src.NTK import GaussianFit
from matplotlib.animation import FuncAnimation
from ipywidgets import interact
import ipywidgets as widgets
from time import time
import matplotlib.pyplot as plt
import random
import pickle

 
def MSELoss_batch(y_hat, y):
    return 0.5*(y_hat-y).pow(2).sum(-1)


def decision_boundary(x: torch.Tensor, treshold: float, flip_chance: float = 0) -> torch.Tensor:
    y = ((x[:, 0] > treshold) | (x[:, 1] > treshold)).float()
    should_flip = torch.rand(y.size()) < flip_chance
    y[should_flip] = 1 - y[should_flip]
    return y

def sample_data(n: int, treshold: float = 0.5**0.5, seed: Optional[int] = None, flip_chance: float = 0) -> tuple[torch.Tensor, torch.Tensor]:
    if seed is not None:
        torch.manual_seed(seed)
    x = torch.rand(n, 2)
    y = decision_boundary(x, treshold, flip_chance)
    return x, y


def plot_decision_boundary(model: torch.nn.Module, treshold: float, fig, ax) -> None:
    model.eval()
    y_hat = torch.stack([model.forward(x_test_batch) for x_test_batch in torch.split(x_test, 100)]).view(-1)
    y_acc = (y_hat >= 0.5)== ((x_test[:, 0] > treshold) | (x_test[:, 1] > treshold))
    # Clear the axis
    ax.clear()
    # clear the colorbar
    # Plot decision boundary
    cb = ax.contourf(x0, x1, y_hat.view(int(N_testing**0.5), int(N_testing**0.5)).detach().numpy().reshape(int(N_testing**0.5), int(N_testing**0.5)), alpha=0.9, levels=torch.linspace(-5.5, 5.5, 10))
    # add colorbar
    # Plot training data
    ax.scatter(x_train[y_train == 0, 0], x_train[y_train == 0, 1], c='r', label='0', s=1, marker='x')
    ax.scatter(x_train[y_train == 1, 0], x_train[y_train == 1, 1], c='b', label='1', s=1, marker='x')
    # Add title
    ax.set_title('Accuracy: {:.2f}'.format(y_acc.float().mean().item()))
    ax.set_xlabel('x0')
    ax.set_ylabel('x1')
    
    # Make striped line to mark the decision boundary (x0 < treshold and x1 < treshold)
    ax.plot([0, treshold], [treshold, treshold], 'k--')
    # Make ax-line
    ax.axvline(x=treshold, color='k', linestyle='--', ymax=treshold)
    model.train()

# Setup data
treshold = 0.5**0.5
flip_chance = 0.0
N_training = 100
N_testing = 10**4
x_train, y_train = sample_data(N_training, seed=0, flip_chance=flip_chance, treshold=treshold)
# Construct grid of sqrt(N_testing) x sqrt(N_testing) points
x = torch.linspace(0, 1, int(N_testing**0.5))
x0, x1 = torch.meshgrid(x, x, indexing='ij')
x_test = torch.stack([x0.flatten(), x1.flatten()], 1)
y_test = decision_boundary(x_test, treshold)

# Construct NN_model
model_arch = SingleLayerMLP
#optimizer = torch.optim.SGD(model.parameters(), lr=0.02, momentum=0.9)
criterion = lambda x, y: ((x-y)**2).mean()  # torch.nn.BCEWithLogitsLoss()

gradient_loader = torch.utils.data.DataLoader([(x_train[i], y_train[i, None]) for i in range(N_training)], batch_size=64, shuffle=False)

# Slider limits
n_hidden_min = 50
n_hidden_max = 4*1024

# Get largest initialisation of parameters of network
model = model_arch(2, 1, n_hidden_max)
max_weights = model.get_weights()
max_biases = model.get_biases()

model.eval()
model = model_arch(2, 1, 512)
model.set_weights(max_weights, max_biases, 1)
optimizer = torch.optim.SGD(model.parameters(), lr=0.02, momentum=0.9)

noise_points = 51
parameter_count = 7
run_experiment = True

if run_experiment:
    noise_levels = np.logspace(-14, 0, noise_points)
    parameter_levels = np.logspace(np.log10(N_training), np.log10(n_hidden_max), parameter_count)

    accuracy = np.zeros((parameter_count, noise_points))
    kernel_vars = np.zeros((parameter_count,))
    for i, parameter_level in enumerate(tqdm(parameter_levels)):
        parameter_level = int(parameter_level)
        model.eval()
        model = model_arch(2, 1, parameter_level)
        model.set_weights(max_weights, max_biases, 1)
        optimizer = torch.optim.SGD(model.parameters(), lr=0.02, momentum=0.9)
        
        for j, noise_level in enumerate(noise_levels):
            kernel_model = GaussianFit(model, "cpu", noise_var=noise_level)
            kernel_model.fit(gradient_loader, optimizer, MSELoss_batch)
            
            y_hat = torch.stack([kernel_model.forward(x_test_batch) for x_test_batch in torch.split(x_test, 100)]).view(-1)
            y_acc = (y_hat >= 0.5)== ((x_test[:, 0] > treshold) | (x_test[:, 1] > treshold))
            accuracy[i, j] = y_acc.float().mean().item()
            
        kernel_vars[i] = kernel_model.covarinace_kernel.diag().mean().cpu()
        
    # Save accuracy, noise_levels and parameter_levels
    with open('noise_kernels.pkl', 'wb') as f:
        pickle.dump((accuracy, noise_levels, parameter_levels, kernel_vars), f)
else:
    with open('noise_kernels.pkl', 'rb') as f:
        accuracy, noise_levels, parameter_levels, kernel_vars = pickle.load(f)


# Plot the accuracy as a function of noise level
colors = plt.cm.viridis(np.linspace(0, 1, len(parameter_levels)))
for i in range(len(parameter_levels)):
    color = colors[i]
    plt.plot(noise_levels, accuracy[i], label=f'Hidden units: {int(parameter_levels[i])}', color=color)
    plt.axvline(kernel_vars[i], color=color, linestyle='--')
# Show variance
plt.legend()
plt.xscale('log')
plt.xlabel('Noise variance')
plt.ylabel('Accuracy')
plt.savefig('noise_kernels.svg')
plt.show()
