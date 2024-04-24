import torch
import torchvision
from src.models.NN_models import SimpleMLP, CNN
import yaml
from datasets.download_CIFAR100 import get_binary_CIFAR100
from src.training import train_epoch, validate, test
from tqdm import tqdm
from easydict import EasyDict as edict
import matplotlib.pyplot as plt

config = edict(yaml.safe_load(open('configs/binary_CIFAR100.yaml', 'r')))
# Define the model (binary classification)
torch.manual_seed(1)
model = CNN(input_shape=(32, 32, 3), output_size=1, hidden_size=config.model.hidden_size, num_layers=config.model.num_layers)
optimizer = torch.optim.Adam(model.parameters(), lr=config.training.learning_rate)
loss = torch.nn.BCEWithLogitsLoss()

# Load the data
digits = (1, 7)
train_data, val_data, test_data = get_binary_CIFAR100(digits, config.training.validation_split)
# Shuffle the train data
train_loader = torch.utils.data.DataLoader(train_data, batch_size=config.training.batch_size, shuffle=True)
val_loader = torch.utils.data.DataLoader(val_data, batch_size=config.training.batch_size, shuffle=True)
test_loader = torch.utils.data.DataLoader(test_data, batch_size=config.training.batch_size, shuffle=True)

n_samples = torch.logspace(1, torch.log10(torch.tensor(2*250)), 10, dtype=int).round().int()
indexes = torch.randperm(len(train_data))

def run_experiment(n_samples: int, unc_sample: bool=False) -> tuple[list[float], list[float]]:
    torch.manual_seed(1)
    model = SimpleMLP(input_size=32*32*3, output_size=1, hidden_size=config.model.hidden_size, num_layers=config.model.num_layers)
    optimizer = torch.optim.Adam(model.parameters(), lr=config.training.learning_rate)
    
    indexed_data = torch.utils.data.Subset(train_data, indexes[:n_samples])
    non_indexed_data = torch.utils.data.Subset(train_data, indexes[n_samples:])
    train_loader = torch.utils.data.DataLoader(indexed_data, batch_size=config.training.batch_size, shuffle=True)
    if len(non_indexed_data) > 0:
        non_train_loader = torch.utils.data.DataLoader(non_indexed_data, batch_size=config.training.batch_size, shuffle=True)
    else:
        non_train_loader = []

    # Train the model
    validation_losses = []
    validation_accuracies = []

    for epoch in range(config.training.num_epochs):
        # Validate model
        val_loss, val_accuracy = validate(model, loss, val_loader)
        validation_losses.append(val_loss)
        validation_accuracies.append(val_accuracy)
        
        train_epoch(model, optimizer, loss, train_loader)
    if unc_sample and len(non_train_loader) > 0:
        # Compute predictions of the non-indexed data
        y_hat = torch.cat([model(x) for x, _ in non_train_loader])
        # Find most uncertain points
        most_unc_points = torch.abs(y_hat[:, 0]).sort().indices
        indexes[n_samples:] = indexes[n_samples:][most_unc_points]
        
    val_loss, val_accuracy = validate(model, loss, val_loader)
    validation_losses.append(val_loss)
    validation_accuracies.append(val_accuracy)
        
    return validation_losses, validation_accuracies

sample_validation_losses = []
sample_validation_accuracy = []
for n in tqdm(n_samples, desc='Running sample sizes for PL'):
    validation_losses, validation_accuracies = run_experiment(n, unc_sample=False)
    sample_validation_losses.append(validation_losses)
    sample_validation_accuracy.append(validation_accuracies)
    
sample_validation_unc_losses = []
sample_validation_unc_accuracy = []
for n in tqdm(n_samples, desc='Running sample sizes for Uncertainty Sampling'):
    validation_losses, validation_accuracies = run_experiment(n, unc_sample=True)
    sample_validation_unc_losses.append(validation_losses)
    sample_validation_unc_accuracy.append(validation_accuracies)
    
# Plot the validation loss and accuracy as a function of the number of samples
fig, axs = plt.subplots(2)
for i, n in enumerate(n_samples):
    axs[0].plot(sample_validation_losses[i], label=f'{n} samples')
    axs[1].plot(sample_validation_accuracy[i], label=f'{n} samples')
axs[0].set_title('Validation Loss')
axs[0].set_xlabel('Epoch')
axs[0].set_ylabel('Loss')
axs[0].legend()

axs[1].set_title('Validation Accuracy')
axs[1].set_xlabel('Epoch')
axs[1].set_ylabel('Accuracy')
axs[1].legend()
plt.savefig("figures/binary_MNIST_sample_runs.svg")
plt.show()

fig, axs = plt.subplots(2)
# Passive learning
final_losses = [sample_validation_losses[i][-1] for i in range(len(n_samples))]
final_accuracies = [sample_validation_accuracy[i][-1] for i in range(len(n_samples))]
min_losses = [min(sample_validation_losses[i]) for i in range(len(n_samples))]
max_acc = [max(sample_validation_accuracy[i]) for i in range(len(n_samples))]

# Uncertainty sampling
final_unc_losses = [sample_validation_unc_losses[i][-1] for i in range(len(n_samples))]
final_unc_accuracies = [sample_validation_unc_accuracy[i][-1] for i in range(len(n_samples))]
min_unc_losses = [min(sample_validation_unc_losses[i]) for i in range(len(n_samples))]
max_unc_acc = [max(sample_validation_unc_accuracy[i]) for i in range(len(n_samples))]

axs[0].plot(n_samples, final_losses, label='Final Loss PL', color='red', linestyle='--')
axs[0].plot(n_samples, min_losses, label='Min Loss PL', color='red')
axs[0].plot(n_samples, final_unc_losses, label='Final Loss unc', color='green', linestyle='--')
axs[0].plot(n_samples, min_unc_losses, label='Min Loss unc', color='green')
axs[0].set_title('Final Loss')
axs[0].set_xlabel('Number of Samples')
axs[0].set_ylabel('Loss')
axs[0].set_xscale('log')
axs[0].legend()

axs[1].plot(n_samples, final_accuracies, label='Final Accuracy PL', color='red', linestyle='--')
axs[1].plot(n_samples, max_acc, label='Max Accuracy PL', color='red')
axs[1].plot(n_samples, final_unc_accuracies, label='Final Accuracy unc', color='green', linestyle='--')
axs[1].plot(n_samples, max_unc_acc, label='Max Accuracy unc', color='green')
axs[1].set_title('Final Accuracy')
axs[1].set_xlabel('Number of Samples')
axs[1].set_ylabel('Accuracy')
axs[1].set_xscale('log')
axs[1].legend()
plt.savefig("figures/binary_MNIST_sample_runs_final.svg")
plt.show()


    
# Test the model
#test_accuracy = test(model, test_loader)
        
#print(f'Test Accuracy: {test_accuracy}')

# Plot the validation loss and accuracy
fig, axs = plt.subplots(2)
axs[0].plot(validation_losses, label='Validation Loss')
axs[0].set_title('Validation Loss')
axs[0].set_xlabel('Epoch')
axs[0].set_ylabel('Loss')

axs[1].plot(validation_accuracies, label='Validation Accuracy')
axs[1].set_title('Validation Accuracy')
axs[1].set_xlabel('Epoch')
axs[1].set_ylabel('Accuracy')
plt.savefig("figures/binary_MNIST.svg")
plt.show()