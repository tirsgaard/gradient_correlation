from typing import Iterable
import torch
import torchvision
import scipy
from src.models.NN_models import SingleLayerMLP
import yaml
from datasets.download_MNIST import get_MNIST
from src.training import train_epoch, validate, test
from src.correlation_gradient import get_gradient, rank_sample_information, rank_correlation_uniqueness, construct_covariance_matrix
from tqdm import tqdm
from easydict import EasyDict as edict
import matplotlib.pyplot as plt
from copy import deepcopy

config = edict(yaml.safe_load(open('configs/MNIST.yaml', 'r')))
device = torch.device("cpu") #torch.device('cuda' if torch.cuda.is_available() else ("mps" if torch.backends.mps.is_available() else 'cpu'))

torch.manual_seed(1)

loss = torch.nn.CrossEntropyLoss()

epochs = 100
# Load the data
train_data, val_data, test_data = get_MNIST(0.1)
gradient_batch_size = 128
# Shuffle the train data
small_training_set = torch.utils.data.Subset(train_data, range(gradient_batch_size))
train_loader = torch.utils.data.DataLoader(small_training_set, batch_size=config.training.batch_size, shuffle=True, pin_memory=True)
val_loader = torch.utils.data.DataLoader(val_data, batch_size=config.validation.batch_size, shuffle=True, pin_memory=True)
gradient_loader = torch.utils.data.DataLoader(small_training_set, batch_size=1, shuffle=False, pin_memory=True)

validation_acc = torch.zeros((3, epochs))

# First 100 hidden units
N_hidden = 500
model = SingleLayerMLP(input_size=784, output_size=10, hidden_size=N_hidden).to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=config.training.learning_rate)
grads = get_gradient(model, gradient_loader, loss, optimizer, True, flatten= True, use_label=True)
covarinace_matrix = construct_covariance_matrix(grads)
u, s, v = scipy.linalg.svd(covarinace_matrix)
eigenvalues1 = []
eigenvalues1.append(s[:5])
print(s[:5])
first = s[:5]
for epoch in tqdm(range(epochs)):
    for j in range(50):
        train_epoch(model, optimizer, loss, train_loader, device)
    val_loss, val_acc = validate(model, loss, val_loader, device)
    print("Epoch: {}, Val Loss: {}, Val Acc: {}".format(epoch, val_loss, val_acc))
    validation_acc[0, epoch] = val_acc
    grads = get_gradient(model, gradient_loader, loss, optimizer, True, flatten= True, use_label=True)
    covarinace_matrix = torch.matmul(grads, grads.T)
    u, s, v = scipy.linalg.svd(covarinace_matrix)
    eigenvalues1.append(s[:5])
    print(s[:5]/first)
    print(s[:5])

# Plot eigenvalues
eigenvalue1 = torch.tensor(eigenvalues1)


# First 1000 hidden units
N_hidden = 1000
model = SingleLayerMLP(input_size=784, output_size=10, hidden_size=N_hidden).to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=config.training.learning_rate)
grads = get_gradient(model, gradient_loader, loss, optimizer, True, flatten= True, use_label=True)
covarinace_matrix = construct_covariance_matrix(grads)
u, s, v = scipy.linalg.svd(covarinace_matrix)
eigenvalues2 = []
eigenvalues2.append(s[:5])
print(s[:5])
first = s[:5]
for epoch in tqdm(range(epochs)):
    for j in range(50):
        train_epoch(model, optimizer, loss, train_loader, device)
    val_loss, val_acc = validate(model, loss, val_loader, device)
    print("Epoch: {}, Val Loss: {}, Val Acc: {}".format(epoch, val_loss, val_acc))
    validation_acc[1, epoch] = val_acc
    grads = get_gradient(model, gradient_loader, loss, optimizer, True, flatten= True, use_label=True)
    covarinace_matrix = torch.matmul(grads, grads.T)
    u, s, v = scipy.linalg.svd(covarinace_matrix)
    eigenvalues2.append(s[:5])
    print(s[:5]/first)
    print(s[:5])

# First 1000 hidden units
N_hidden = 10000
model = SingleLayerMLP(input_size=784, output_size=10, hidden_size=N_hidden).to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=config.training.learning_rate)
grads = get_gradient(model, gradient_loader, loss, optimizer, True, flatten= True, use_label=True)
covarinace_matrix = construct_covariance_matrix(grads)
u, s, v = scipy.linalg.svd(covarinace_matrix)
eigenvalues3 = []
eigenvalues3.append(s[:5])
print(s[:5])
first = s[:5]
for epoch in tqdm(range(epochs)):
    for j in range(50):
        train_epoch(model, optimizer, loss, train_loader, device)
    val_loss, val_acc = validate(model, loss, val_loader, device)
    print("Epoch: {}, Val Loss: {}, Val Acc: {}".format(epoch, val_loss, val_acc))
    validation_acc[2, epoch] = val_acc
    grads = get_gradient(model, gradient_loader, loss, optimizer, True, flatten= True, use_label=True)
    covarinace_matrix = torch.matmul(grads, grads.T)
    u, s, v = scipy.linalg.svd(covarinace_matrix)
    eigenvalues3.append(s[:5])
    print(s[:5]/first)
    print(s[:5])


fig, axs = plt.subplots(2, 1) 
# Plot eigenvalues
eigenvalue2 = torch.tensor(eigenvalues2)
axs[0].plot(eigenvalues1, label = 'N_hidden = 500', color='yellow')
axs[0].plot(eigenvalues2, label = 'N_hidden = 1 000', color='orange')
axs[0].plot(eigenvalues3, label = 'N_hidden = 10 000', color='red')
#plt.yscale('log')
axs[0].legend()

# Plot accuracy
axs[1].plot(validation_acc[0], label = 'N_hidden = 500', color='yellow')
axs[1].plot(validation_acc[1], label = 'N_hidden = 1 000', color='orange')
axs[1].plot(validation_acc[2], label = 'N_hidden = 10 000', color='red')
axs[1].legend()
plt.show()
