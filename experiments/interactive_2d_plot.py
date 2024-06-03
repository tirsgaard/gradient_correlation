import torch
import numpy as np
import matplotlib.pyplot as plt
from typing import Optional

from src.models.NN_models import SimpleMLP, SingleLayerMLP
from src.NTK import GaussianFit
from matplotlib.animation import FuncAnimation
from ipywidgets import interact
import ipywidgets as widgets
from time import time
import matplotlib.pyplot as plt
import random
 
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
flip_chance = 0.3
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
n_hidden_max = 8*1024


# Get largest initialisation of parameters of network
model = model_arch(2, 1, n_hidden_max)
max_weights = model.get_weights()
max_biases = model.get_biases()

def plot_NTK_decision_boundary(n_hidden: int = 64, initial_gain: float = 1.0, noise_var: float = 0.0):
    #t = time()
    n_hidden = int(n_hidden)
    model = model_arch(2, 1, n_hidden)
    model.set_weights(max_weights, max_biases, initial_gain)
    
    optimizer = torch.optim.SGD(model.parameters(), lr=0.02, momentum=0.9)
    kernel_model = GaussianFit(model, "cpu", noise_var=noise_var)
    kernel_model.fit(gradient_loader, optimizer, MSELoss_batch)
    fig, ax = plt.subplots()
    plot_decision_boundary(kernel_model, treshold, fig, ax)
    #print(f'Time: {time()-t}')
    plt.show()

n_hidden_slider = widgets.FloatLogSlider(value=(np.log10(n_hidden_min) + np.log10(n_hidden_max))/2, base=10, min=np.log10(n_hidden_min), max=np.log10(n_hidden_max), step=0.01, description='n_hidden')
initial_gain = widgets.FloatSlider(value=1, min=0, max=50, step=0.01, description='initial gain')
noise_var = widgets.FloatLogSlider(value=-5, base=10, min=np.log10(10**-14), max=np.log10(10**0), step=0.001, description='Noise variance')

interact(plot_NTK_decision_boundary, n_hidden=n_hidden_slider, initial_gain=initial_gain, noise_var=noise_var)
"""
global epoch
epoch = 0

# creating the first plot and frame
fig, axs = plt.subplots(1, 2)
plot_decision_boundary(kernel_model, treshold, fig, axs[0])

ax = axs[1]
# Fit the model in the limit using NTK
ax.set_xlim(0, 1)
ax.set_ylim(0, 1)

def train_epoch(frame):
    global epoch
    optimizer.zero_grad()
    y_hat = model(x_train)
    l = criterion(y_hat, y_train.view(-1, 1))
    l.backward()
    optimizer.step()
    
    if epoch % 10 == 0:
        print(f'Loss: {l.item()}')
        plot_decision_boundary(model, treshold, fig, ax)
    epoch += 1

 
anim = FuncAnimation(fig, train_epoch, frames = None)
plt.show()
"""