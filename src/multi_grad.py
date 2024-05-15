import torch
from datasets.download_MNIST import get_MNIST
from src.models.NN_models import SingleLayerMLP, SimpleMLP
from time import time
import opacus

def MSELoss(y_hat, y):
    return 0.5*(y_hat-y).pow(2).sum(-1)

# Get data
train_data, val_data, test_data = get_MNIST(0.1)

# Define the model
device = torch.device("mps")
model = SingleLayerMLP(input_size=784, output_size=10, hidden_size=500).to(device)

# Define the optimizer
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

# Define the loss function
loss = MSELoss

# Make a single batch
gradient_batch_size = 256
small_training_set = torch.utils.data.Subset(train_data, range(gradient_batch_size))
train_loader = torch.utils.data.DataLoader(small_training_set, batch_size=gradient_batch_size, shuffle=False)

x, y = next(iter(train_loader))
x = x.to(device)
y = y.to(device)
n = 100

# Compute combined gradient for model
overall_time = time()
for j in range(n):
    overall_grads = []
    losses = loss(model(x), y)
    for i in range(gradient_batch_size):
        optimizer.zero_grad()
        losses[i].backward(retain_graph=True)
        overall_grads.append(torch.cat([param.grad.view(-1) for param in model.parameters()]))
overall_grads = torch.stack(overall_grads)
print(f"Overall Time: {time()-overall_time}")

# Compute individual gradients at the same time using the jacobian product
individual_time = time()
for j in range(n):
    optimizer.zero_grad()
    losses = loss(model(x), y)
    grads = torch.autograd.grad(losses, model.parameters(), is_grads_batched=True, grad_outputs=torch.eye(gradient_batch_size).to(device))
    #grads = torch.autograd.grad(losses.split(1), [model.parameters()]*gradient_batch_size)
    individual_grads = torch.cat([grad.view(gradient_batch_size, -1) for grad in grads], 1)
print(f"Individual Time: {time()-individual_time}")
# Check if the two gradients are the same
assert torch.isclose(overall_grads, individual_grads).float().mean() >=0.999

# Compute individual gradients at the same time using the opacus library

# Wrap the model in a privacy engine
from opacus.grad_sample import GradSampleModule
# Wrap each module in model in a GradSampleModule
import torch.nn as nn
model.layers = nn.ModuleList([GradSampleModule(layer) for layer in model.layers])
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
privacy_time = time()
for j in range(n):
    # Compute the gradients
    optimizer.zero_grad()
    for parm in model.parameters():
        parm.grad_sample = None
    losses = loss(model(x), y).mean()
    # Get gradients
    losses.backward()
    opacus_grads = torch.cat([grad.grad_sample.view(gradient_batch_size, -1) for grad in model.parameters()], 1)
    
print(f"Privacy Time: {time()-privacy_time}")
# Check if the two gradients are the same
assert torch.isclose(overall_grads, opacus_grads).float().mean() >=0.999    

