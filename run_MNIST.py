import torch
import torchvision
from models.NN_models import SimpleMLP
import yaml
from datasets.download_MNIST import get_binary_MNIST
from training import train_epoch, validate, test
from tqdm import tqdm
from easydict import EasyDict as edict
import matplotlib.pyplot as plt

config = edict(yaml.safe_load(open('configs/binary_MNIST.yaml', 'r')))
# Define the model (binary classification)
torch.manual_seed(1)
model = SimpleMLP(input_size=784, output_size=1, hidden_size=config.model.hidden_size, num_layers=config.model.num_layers)
optimizer = torch.optim.Adam(model.parameters(), lr=config.training.learning_rate)
loss = torch.nn.BCEWithLogitsLoss()

# Load the data
digits = (1, 7)
train_data, val_data, test_data = get_binary_MNIST(digits, config.training.validation_split)
# Shuffle the train data
train_loader = torch.utils.data.DataLoader(train_data, batch_size=config.training.batch_size, shuffle=True)
val_loader = torch.utils.data.DataLoader(val_data, batch_size=config.training.batch_size, shuffle=True)
test_loader = torch.utils.data.DataLoader(test_data, batch_size=config.training.batch_size, shuffle=True)

n_samples = torch.logspace(1, 4, 100, dtype=int).round().int()
indexes = torch.randperm(len(train_data))
sample_validation_losses = []
sample_validation_accuracy = []
for n in tqdm(n_samples, desc='Running sample sizes'):
    torch.manual_seed(1)
    model = SimpleMLP(input_size=784, output_size=1, hidden_size=config.model.hidden_size, num_layers=config.model.num_layers)
    optimizer = torch.optim.Adam(model.parameters(), lr=config.training.learning_rate)
    
    indexed_data = torch.utils.data.Subset(train_data, indexes[:n])
    non_indexed_data = torch.utils.data.Subset(train_data, indexes[n:])
    train_loader = torch.utils.data.DataLoader(indexed_data, batch_size=config.training.batch_size, shuffle=True)
    #non_train_loader = torch.utils.data.DataLoader(non_indexed_data, batch_size=config.training.batch_size, shuffle=True)

    # Train the model
    validation_losses = []
    validation_accuracies = []

    for epoch in range(config.training.num_epochs):
        # Validate model
        val_loss, val_accuracy = validate(model, loss, val_loader)
        validation_losses.append(val_loss)
        validation_accuracies.append(val_accuracy)
        #print(f'Epoch {epoch}, val loss: {val_loss}, val accuracy: {val_accuracy}')
        
        train_epoch(model, optimizer, loss, train_loader)
        
    # Validate model
    val_loss, val_accuracy = validate(model, loss, val_loader)
    validation_losses.append(val_loss)
    validation_accuracies.append(val_accuracy)
    sample_validation_losses.append(validation_losses)
    sample_validation_accuracy.append(validation_accuracies)
    
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
final_losses = [sample_validation_losses[i][-1] for i in range(len(n_samples))]
final_accuracies = [sample_validation_accuracy[i][-1] for i in range(len(n_samples))]
min_losses = [min(sample_validation_losses[i]) for i in range(len(n_samples))]
max_acc = [max(sample_validation_accuracy[i]) for i in range(len(n_samples))]

axs[0].plot(n_samples, final_losses, label='Final Loss')
axs[0].plot(n_samples, min_losses, label='Min Loss')
axs[0].set_title('Final Loss')
axs[0].set_xlabel('Number of Samples')
axs[0].set_ylabel('Loss')
axs[0].set_xscale('log')

axs[1].plot(n_samples, final_accuracies, label='Final Accuracy')
axs[1].plot(n_samples, max_acc, label='Max Accuracy')
axs[1].set_title('Final Accuracy')
axs[1].set_xlabel('Number of Samples')
axs[1].set_ylabel('Accuracy')
axs[1].set_xscale('log')
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