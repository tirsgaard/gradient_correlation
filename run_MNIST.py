import torch
import torchvision
from models.NN_models import SimpleMLP
import yaml
from datasets.download_MNIST import get_MNIST_train, get_MNIST_test
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
train_data = get_MNIST_train()
test_data = get_MNIST_test()

# Only keep the 0s and 1s for binary classification
digits = (1,7)
train_data = list(filter(lambda i: i[1] in digits, train_data))
test_data = list(filter(lambda i: i[1] in digits, test_data))
# Set 0 to be the negative class
train_data = [(x, 0 if y == digits[0] else 1) for x, y in train_data]
test_data = [(x, 0 if y == digits[0] else 1) for x, y in test_data]

# Split the training set into training and validation
val_split = 0.2
train_size = int((1 - val_split) * len(train_data))
val_size = len(train_data) - train_size
train_data, val_data = torch.utils.data.random_split(train_data, [train_size, val_size])
train_loader = torch.utils.data.DataLoader(train_data, batch_size=config.training.batch_size, shuffle=True)
val_loader = torch.utils.data.DataLoader(val_data, batch_size=config.training.batch_size, shuffle=True)
test_loader = torch.utils.data.DataLoader(test_data, batch_size=config.training.batch_size, shuffle=True)

# Train the model
validation_losses = []
validation_accuracies = []

for epoch in tqdm(range(config.training.num_epochs), desc='Training Epochs'):
    # Validate
    model.eval()
    val_loss = 0
    val_accuracy = 0
    with torch.no_grad():
        for x, y in val_loader:
            y_hat = model(x)
            l = loss(y_hat, y.float().view(-1, 1))
            y_hat = (y_hat > 0).float()
            val_accuracy += (y_hat == y.float().view(-1, 1)).sum().item()
            val_loss += l.item()
    val_loss /= len(val_loader)
    val_accuracy /= len(val_data)
    validation_losses.append(val_loss)
    validation_accuracies.append(val_accuracy)
    print(f'Epoch {epoch}, val loss: {val_loss}, val accuracy: {val_accuracy}')
    
    model.train()
    for i, (x, y) in enumerate(train_loader):
        optimizer.zero_grad()
        y_hat = model(x)
        l = loss(y_hat, y.float().view(-1, 1))
        l.backward()
        optimizer.step()
    

    
# Test the model
model.eval()
correct = 0
total = 0
with torch.no_grad():
    for x, y in tqdm(test_loader, desc='Testing model'):
        y_hat = model(x)
        y_hat = (y_hat > 0).float()
        correct += (y_hat == y.float().view(-1, 1)).sum().item()
        total += y.size(0)
        
print(f'Test Accuracy: {correct / total}')

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