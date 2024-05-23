from typing import Iterable
import torch
import torch.nn.functional as F
import torchvision
import scipy
from src.models.NN_models import SimpleMLP, SingleLayerMLP
import yaml
from datasets.download_MNIST import get_MNIST
from src.training import train_epoch, validate, test
from src.correlation_gradient import rank_sample_information, rank_correlation_uniqueness, rank_uncertainty_information
from src.utility import JS_div
from tqdm import tqdm
from easydict import EasyDict as edict
import matplotlib.pyplot as plt
from copy import deepcopy
import ivon

config = edict(yaml.safe_load(open('configs/MNIST.yaml', 'r')))
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

torch.manual_seed(1)
model = SimpleMLP(input_size=784, output_size=10, hidden_size=config.model.hidden_size, num_layers=config.model.num_layers).to(device)

model_copy = deepcopy(model)

#torch.optim.Adam(model.parameters(), lr=config.training.learning_rate)
loss = torch.nn.CrossEntropyLoss() #lambda x, y: 0.5*(x-y).pow(2).sum(-1).mean()  # 
loss_batched = torch.nn.CrossEntropyLoss(reduction='none')

# Load the data
train_data, val_data, test_data = get_MNIST(config.training.validation_split, device=device)
train_data = torch.utils.data.Subset(train_data, torch.randperm(len(train_data))[:1000])
# Shuffle the train data
train_loader = torch.utils.data.DataLoader(train_data, batch_size=16, shuffle=True, pin_memory=True)
val_loader = torch.utils.data.DataLoader(val_data, batch_size=config.training.batch_size, shuffle=True, pin_memory=True)
test_loader = torch.utils.data.DataLoader(test_data, batch_size=config.validation.batch_size, shuffle=True, pin_memory=True)
train_samples = 1
test_samples = 10
momentum = 0.9
weight_decay = 1e-6
h0 = 0.01

optimizer = ivon.IVON(model.parameters(), lr=0.01, ess=len(train_data), beta1=momentum, weight_decay=weight_decay)
# Train model
for epoch in tqdm(range(100)):
        for i, (x, y) in enumerate(train_loader):
            for _ in range(train_samples):
                with optimizer.sampled_params(train=True):                    
                    optimizer.zero_grad()
                    x, y = x.to(device), y.to(device)
                    y_hat = model(x)
                    l = loss(y_hat, y.float().squeeze(-1))
                    l.backward()
                optimizer.step()
                    

# Validate using the IVON method
model.eval()
validation_acc_ivon = []
prop_of_pred_ivon = []
for (x, y) in val_loader:
    sampled_probs = []
    for i in range(test_samples):
        with optimizer.sampled_params():
            sampled_logit = model(x)
            sampled_probs.append(F.softmax(sampled_logit, dim=1))
    JS_divergence = JS_div(torch.stack(sampled_probs).permute((1, 0, 2)))
    prob = torch.mean(torch.stack(sampled_probs), dim=0)
    prop_of_pred_ivon.append(prob.max(1).values)
    _, prediction = prob.max(1)
    correct = (prediction == y.argmax(1))
    validation_acc_ivon.append(correct)
    
# Stack the probabilities
prop_of_pred_ivon = torch.cat(prop_of_pred_ivon, dim=0)
correct_pred_ivon = torch.cat(validation_acc_ivon, dim=0)
print(f"Validation Accuracy: {sum(correct_pred_ivon)/len(val_loader.dataset)}")

# Train the same model using Adam
model = model_copy
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
for epoch in tqdm(range(100)):
        for i, (x, y) in enumerate(train_loader):
            optimizer.zero_grad()
            x, y = x.to(device), y.to(device)
            y_hat = model(x)
            l = loss(y_hat, y.float().squeeze(-1))
            l.backward()
            optimizer.step()
            
# validate the model
model.eval()
validation_acc_adam = []
prop_of_pred_adam = []
for (x, y) in val_loader:
    sampled_probs = []
    pred = model(x)
    prob = F.softmax(pred, dim=1)
    prop_of_pred_adam.append(prob.max(1).values)
    _, prediction = prob.max(1)
    correct = (prediction == y.argmax(1))
    validation_acc_adam.append(correct)
    
# Stack the probabilities
prop_of_pred_adam = torch.cat(prop_of_pred_adam, dim=0)
correct_pred_adam = torch.cat(validation_acc_adam, dim=0)
print(f"Validation Accuracy: {sum(correct_pred_adam)/len(val_loader.dataset)}")

# Make a calibration plot

# Make a calibration plot
bins = torch.linspace(0, 1, 10)
bin_indices = torch.bucketize(prop_of_pred_ivon, bins)
calibration_ivon = torch.zeros(len(bins))
for i in range(len(bins)):
    calibration_ivon[i] = torch.mean(correct_pred_ivon[bin_indices == i].float())

bin_indices = torch.bucketize(prop_of_pred_adam, bins)
calibration = torch.zeros(len(bins))
for i in range(len(bins)):
    calibration[i] = torch.mean(correct_pred_adam[bin_indices == i].float())

plt.plot(bins, calibration, label="Adam")
plt.plot(bins, calibration_ivon, label="IVON")
plt.plot(bins, bins, linestyle='--', color='black', label="Perfect Calibration")
plt.xlabel("Predicted Probability")
plt.ylabel("Accuracy")
plt.title("Calibration Plot")
plt.legend()
plt.show()