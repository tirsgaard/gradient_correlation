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
digits = (2,4)
train_data, val_data, test_data = get_binary_MNIST(digits, config.training.validation_split)
train_loader = torch.utils.data.DataLoader(train_data, batch_size=config.training.batch_size, shuffle=True)
val_loader = torch.utils.data.DataLoader(val_data, batch_size=config.training.batch_size, shuffle=True)
test_loader = torch.utils.data.DataLoader(test_data, batch_size=config.training.batch_size, shuffle=True)

# Train the model
validation_losses = []
validation_accuracies = []

for epoch in tqdm(range(config.training.num_epochs), desc='Training Epochs'):
    # Validate model
    val_loss, val_accuracy = validate(model, loss, val_loader)
    validation_losses.append(val_loss)
    validation_accuracies.append(val_accuracy)
    print(f'Epoch {epoch}, val loss: {val_loss}, val accuracy: {val_accuracy}')
    
    train_epoch(model, optimizer, loss, train_loader)
    
# Validate model
val_loss, val_accuracy = validate(model, loss, val_loader)
validation_losses.append(val_loss)
validation_accuracies.append(val_accuracy)
print(f'Epoch {epoch}, val loss: {val_loss}, val accuracy: {val_accuracy}')
    
# Test the model
test_accuracy = test(model, test_loader)
        
print(f'Test Accuracy: {test_accuracy}')

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