from typing import Iterable
import torch
import torchvision
from src.models.NN_models import SimpleMLP
import yaml
from datasets.download_MNIST import get_MNIST
from src.training import train_epoch, validate, test
from src.correlation_gradient import rank_sample_information
from tqdm import tqdm
from easydict import EasyDict as edict
import matplotlib.pyplot as plt
from copy import deepcopy

config = edict(yaml.safe_load(open('configs/MNIST.yaml', 'r')))
device = torch.device("cpu") #torch.device('cuda' if torch.cuda.is_available() else ("mps" if torch.backends.mps.is_available() else 'cpu'))

torch.manual_seed(1)
model = SimpleMLP(input_size=784, output_size=10, hidden_size=config.model.hidden_size, num_layers=config.model.num_layers).to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=config.training.learning_rate)
loss = torch.nn.CrossEntropyLoss()

# Load the data
train_data, val_data, test_data = get_MNIST(config.training.validation_split)
# Shuffle the train data
train_loader = torch.utils.data.DataLoader(train_data, batch_size=config.training.batch_size, shuffle=True, pin_memory=True)
val_loader = torch.utils.data.DataLoader(val_data, batch_size=config.training.batch_size, shuffle=True, pin_memory=True)
test_loader = torch.utils.data.DataLoader(test_data, batch_size=config.training.batch_size, shuffle=True, pin_memory=True)

n_samples = torch.logspace(1, 2, 2, dtype=int).round().int()
indexes = torch.randperm(len(train_data))

def uncertainty_rank_datapoints(data_loader: torch.utils.data.DataLoader, model: torch.nn.Module) -> torch.Tensor:
    """ Rank the data points by uncertainty 
    Args:
        data_loader: the data loader to use
        model: the model to use
        
    Returns:
        most_unc_points: the indices of the most uncertain points
    """
    device = next(model.parameters()).device
    # Compute predictions of the non-indexed data
    y_hat = torch.cat([model(x.to(device)) for x, _ in data_loader])
    # Find most uncertain points
    most_unc_points = y_hat.softmax(-1).max(-1)[0].sort().indices
    return most_unc_points
    
    
def run_single_experiment(sampled_indexes: torch.Tensor, unsampled_indexes: torch.Tensor, n_additional_samples: Iterable, max_add_subset: int = 1000) -> tuple[list[float], list[float], list[float]]:
    """ Run an active learning experiment. 
    Args:
        sampled_indexes: the indexes to use for training
        unsampled_indexes: the indexes to select new datapoints from
        n_additional_samples: the number of samples to try to add to the training set
    
    Returns:
        random_sampling_results: the results for random sampling
        uncertainty_sampling_results: the results for uncertainty sampling
        gradient_correlation_results: the results for gradient correlation sampling
    
    """
    model = SimpleMLP(input_size=784, output_size=10, hidden_size=config.model.hidden_size, num_layers=config.model.num_layers).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=config.training.learning_rate)

    # Split the data    
    indexed_data = torch.utils.data.Subset(train_data, sampled_indexes)
    non_indexed_data = torch.utils.data.Subset(train_data, unsampled_indexes[:max_add_subset])
    train_loader = torch.utils.data.DataLoader(indexed_data, batch_size=config.training.batch_size, shuffle=True, pin_memory=True)
    non_train_loader = torch.utils.data.DataLoader(non_indexed_data, batch_size=config.training.batch_size, shuffle=False, pin_memory=True)
    non_train_loader_single_batch = torch.utils.data.DataLoader(indexed_data + non_indexed_data, batch_size=1, shuffle=False, pin_memory=True)

    # Train the model
    validation_losses = []
    validation_accuracies = []

    for epoch in range(config.training.num_epochs):
        # Validate model
        val_loss, val_accuracy = validate(model, loss, val_loader, device)
        validation_losses.append(val_loss)
        validation_accuracies.append(val_accuracy)
        
        train_epoch(model, optimizer, loss, train_loader, device)
    
    # Get rankings for different sampling strategies
    passive_sampling_ranking = unsampled_indexes[torch.randperm(len(unsampled_indexes))]
    
    # Uncertainty sampling
    most_unc_points = uncertainty_rank_datapoints(non_train_loader, model)
    uncertainty_sampling_ranking = unsampled_indexes[most_unc_points.cpu()]
        
    # Gradient correlation sampling
    most_decor_points = rank_sample_information(non_train_loader_single_batch, model, loss, optimizer, pre_condition_index=torch.tensor(range(len(indexed_data)), dtype=int))
    gradient_correlation_ranking = unsampled_indexes[most_decor_points]
    
    random_sampling_results = []
    uncertainty_sampling_results = []
    gradient_correlation_results = []
    for n_samples in tqdm(n_additional_samples, desc='Running sample sizes'):
        # Random sampling
        random_sampling_data = indexed_data + torch.utils.data.Subset(train_data, passive_sampling_ranking[:n_samples])
        train_data_loader = torch.utils.data.DataLoader(random_sampling_data, batch_size=config.training.batch_size, shuffle=True, pin_memory=True, )
        current_model = deepcopy(model)
        current_optimizer = torch.optim.Adam(current_model.parameters(), lr=config.training.learning_rate)
        random_sampling_results_epoch = []
        for epoch in range(config.training.num_epochs):
            train_epoch(current_model, current_optimizer, loss, train_data_loader, device)
        random_sampling_results_epoch.append(validate(current_model, loss, val_loader, device))
        random_sampling_results.append(list(zip(*random_sampling_results_epoch)))
        
        # Uncertainty sampling
        uncertainty_sampling_data = indexed_data + torch.utils.data.Subset(train_data, uncertainty_sampling_ranking[:n_samples])
        train_data_loader = torch.utils.data.DataLoader(uncertainty_sampling_data, batch_size=config.training.batch_size, shuffle=True, pin_memory=True, )
        current_model = deepcopy(model)
        current_optimizer = torch.optim.Adam(current_model.parameters(), lr=config.training.learning_rate)
        uncertainty_sampling_results_epoch = []
        for epoch in range(config.training.num_epochs):
            train_epoch(current_model, current_optimizer, loss, train_data_loader, device)
        uncertainty_sampling_results_epoch.append(validate(current_model, loss, val_loader, device))
        uncertainty_sampling_results.append(list(zip(*uncertainty_sampling_results_epoch)))
            
        # Gradient correlation sampling
        gradient_correlation_data = indexed_data + torch.utils.data.Subset(train_data, gradient_correlation_ranking[:n_samples])
        train_data_loader = torch.utils.data.DataLoader(gradient_correlation_data, batch_size=config.training.batch_size, shuffle=True, )
        current_model = deepcopy(model)
        current_optimizer = torch.optim.Adam(current_model.parameters(), lr=config.training.learning_rate)
        gradient_correlation_results_epoch = []
        for epoch in range(config.training.num_epochs):
            train_epoch(current_model, current_optimizer, loss, train_data_loader, device)
        gradient_correlation_results_epoch.append(validate(current_model, loss, val_loader, device))
        gradient_correlation_results.append(list(zip(*gradient_correlation_results_epoch)))
        
    return random_sampling_results, uncertainty_sampling_results, gradient_correlation_results

def run_sequence_experiment(indexes: torch.Tensor, n_samples: int, unc_sample: bool=False, corr_sample: bool=False, N_max: int = 100) -> tuple[list[float], list[float]]:
    """ Run an active learning experiment. 
    WARNING that the indexes tensor is modified in place.
    Args:
        indexes: the indexes to use
        n_samples: the number of samples to use
        unc_sample: whether to use uncertainty sampling
        corr_sample: whether to use correlation sampling
        N_max: the maximum number of samples to use for correlation sampling
        
        Returns:
            validation_losses: the validation losses
            validation_accuracies: the validation accuracies
    """
    torch.manual_seed(1)
    model = SimpleMLP(input_size=784, output_size=10, hidden_size=config.model.hidden_size, num_layers=config.model.num_layers)
    optimizer = torch.optim.Adam(model.parameters(), lr=config.training.learning_rate)
    
    indexed_data = torch.utils.data.Subset(train_data, indexes[:n_samples])
    max_samples = min(len(indexes), n_samples + N_max)
    non_indexed_data = torch.utils.data.Subset(train_data, indexes[:max_samples] if corr_sample else indexes[n_samples:max_samples])
    train_loader = torch.utils.data.DataLoader(indexed_data, batch_size=config.training.batch_size, shuffle=True, pin_memory=True)
    if len(non_indexed_data) > 0:
        non_train_loader = torch.utils.data.DataLoader(non_indexed_data, batch_size=config.training.batch_size if not corr_sample else 1, shuffle=False, pin_memory=True)
    else:
        non_train_loader = []

    # Train the model
    validation_losses = []
    validation_accuracies = []

    for epoch in range(config.training.num_epochs):
        # Validate model
        val_loss, val_accuracy = validate(model, loss, val_loader, device)
        validation_losses.append(val_loss)
        validation_accuracies.append(val_accuracy)
        train_epoch(model, optimizer, loss, train_loader, device)
    if unc_sample and len(non_train_loader) > 0:
        # Compute predictions of the non-indexed data
        most_unc_points = uncertainty_rank_datapoints(non_train_loader, model)
        indexes[n_samples:max_samples] = indexes[n_samples:max_samples][most_unc_points]
        
    if corr_sample and len(non_train_loader) > 0:
        most_unc_points = rank_sample_information(non_train_loader, model, loss, optimizer, pre_condition_index=torch.tensor(range(n_samples)))
        indexes[:max_samples] = indexes[:max_samples][most_unc_points]
        
    val_loss, val_accuracy = validate(model, loss, val_loader, device)
    validation_losses.append(val_loss)
    validation_accuracies.append(val_accuracy)
        
    return validation_losses, validation_accuracies

"""
sample_validation_corr_losses = []
sample_validation_corr_accuracy = []
AL_index = indexes.clone()  # The indexes are changed in the run_experiment function
for n in tqdm(n_samples, desc='Running sample sizes for Correlation Sampling'):
    validation_losses, validation_accuracies = run_sequence_experiment(AL_index, n.item(), corr_sample=True)
    sample_validation_corr_losses.append(validation_losses)
    sample_validation_corr_accuracy.append(validation_accuracies)

sample_validation_losses = []
sample_validation_accuracy = []
for n in tqdm(n_samples, desc='Running sample sizes for PL'):
    validation_losses, validation_accuracies = run_sequence_experiment(indexes, n.item(), unc_sample=False)
    sample_validation_losses.append(validation_losses)
    sample_validation_accuracy.append(validation_accuracies)
    
sample_validation_unc_losses = []
sample_validation_unc_accuracy = []
AL_index = indexes.clone()  # The indexes are changed in the run_experiment function
for n in tqdm(n_samples, desc='Running sample sizes for Uncertainty Sampling'):
    validation_losses, validation_accuracies = run_sequence_experiment(AL_index, n.item(), unc_sample=True)
    sample_validation_unc_losses.append(validation_losses)
    sample_validation_unc_accuracy.append(validation_accuracies)
"""
n_start_data = 100
n_rep = 10
results = []
for i in range(n_rep):
    indexes = torch.randperm(len(train_data))
    results.append(run_single_experiment(indexes[:n_start_data], indexes[n_start_data:], n_samples))
    
result_array = torch.tensor(results).mean(0)
sample_validation_losses = result_array[0, :, 0, :]
sample_validation_accuracy = result_array[0, :, 1, :]
sample_validation_unc_losses = result_array[1, :, 0, :]
sample_validation_unc_accuracy = result_array[1, :, 1, :]
sample_validation_corr_losses = result_array[2, :, 0, :]
sample_validation_corr_accuracy = result_array[2, :, 1, :]


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

# Correlation Sampling
final_corr_losses = [sample_validation_corr_losses[i][-1] for i in range(len(n_samples))]
final_corr_accuracies = [sample_validation_corr_accuracy[i][-1] for i in range(len(n_samples))]
min_corr_losses = [min(sample_validation_corr_losses[i]) for i in range(len(n_samples))]
max_corr_acc = [max(sample_validation_corr_accuracy[i]) for i in range(len(n_samples))]

axs[0].plot(n_samples, final_losses, label='Final Loss PL', color='red', linestyle='--')
axs[0].plot(n_samples, min_losses, label='Min Loss PL', color='red')
axs[0].plot(n_samples, final_unc_losses, label='Final Loss unc', color='green', linestyle='--')
axs[0].plot(n_samples, min_unc_losses, label='Min Loss unc', color='green')
axs[0].plot(n_samples, final_corr_losses, label='Final Loss corr', color='blue', linestyle='--')
axs[0].plot(n_samples, min_corr_losses, label='Min Loss corr', color='blue')
axs[0].set_title('Final Loss')
axs[0].set_xlabel('Number of Samples')
axs[0].set_ylabel('Loss')
axs[0].set_xscale('log')
axs[0].legend()

axs[1].plot(n_samples, final_accuracies, label='Final Accuracy PL', color='red', linestyle='--')
axs[1].plot(n_samples, max_acc, label='Max Accuracy PL', color='red')
axs[1].plot(n_samples, final_unc_accuracies, label='Final Accuracy unc', color='green', linestyle='--')
axs[1].plot(n_samples, max_unc_acc, label='Max Accuracy unc', color='green')
axs[1].plot(n_samples, final_corr_accuracies, label='Final Accuracy corr', color='blue', linestyle='--')
axs[1].plot(n_samples, max_corr_acc, label='Max Accuracy corr', color='blue')
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
"""
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
"""