from typing import Iterable
import torch
import torchvision
import scipy
from src.models.NN_models import SimpleMLP
import yaml
from datasets.download_MNIST import get_MNIST
from src.training import train_epoch, validate, test
from src.correlation_gradient import rank_sample_information, rank_correlation_uniqueness
from tqdm import tqdm
from easydict import EasyDict as edict
import matplotlib.pyplot as plt
from copy import deepcopy

config = edict(yaml.safe_load(open('configs/MNIST.yaml', 'r')))
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

torch.manual_seed(1)
model = SimpleMLP(input_size=784, output_size=10, hidden_size=config.model.hidden_size, num_layers=config.model.num_layers).to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=config.training.learning_rate)
loss = torch.nn.CrossEntropyLoss() #lambda x, y: 0.5*(x-y).pow(2).sum(-1).mean()  # 

# Load the data
train_data, val_data, test_data = get_MNIST(config.training.validation_split)
# Shuffle the train data
train_loader = torch.utils.data.DataLoader(train_data, batch_size=config.training.batch_size, shuffle=True, pin_memory=True)
val_loader = torch.utils.data.DataLoader(val_data, batch_size=1, shuffle=True, pin_memory=True)
test_loader = torch.utils.data.DataLoader(test_data, batch_size=config.validation.batch_size, shuffle=True, pin_memory=True)

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
    train_loader_single_batch = torch.utils.data.DataLoader(indexed_data, batch_size=1, shuffle=False, pin_memory=True)
    non_train_loader_single_batch = torch.utils.data.DataLoader(non_indexed_data, batch_size=1, shuffle=False, pin_memory=True)
    comb_train_loader_single_batch = torch.utils.data.DataLoader(indexed_data + non_indexed_data, batch_size=1, shuffle=False, pin_memory=True)

    # Train the model
    validation_losses = []
    validation_accuracies = []
    
    for epoch in range(config.training.num_epochs):
        # Validate model
        val_loss, val_accuracy = validate(model, loss, val_loader, device)
        validation_losses.append(val_loss)
        validation_accuracies.append(val_accuracy)
        for j in range(10):
            train_epoch(model, optimizer, loss, train_loader, device)
    
    # Get rankings for different sampling strategies
    passive_sampling_ranking = unsampled_indexes[torch.randperm(len(unsampled_indexes))]
    
    # Uncertainty sampling
    most_unc_points = uncertainty_rank_datapoints(non_train_loader, model)
    uncertainty_sampling_ranking = unsampled_indexes[most_unc_points.cpu()]
        
    # Gradient correlation sampling
    most_decor_points = rank_sample_information(non_train_loader_single_batch, model, loss, optimizer, pre_condition_index=torch.tensor([], dtype=int))
    gradient_correlation_ranking = unsampled_indexes[most_decor_points]
    
    # Gradient correlation sampling with conditioning on training data
    most_decor_points = rank_sample_information(comb_train_loader_single_batch, model, loss, optimizer, pre_condition_index=torch.tensor(range(len(indexed_data)), dtype=int))
    gradient_correlation_cond_ranking = unsampled_indexes[most_decor_points]
    
    # Gradient correlation sampling with correlation to training data gradient
    most_decor_points = rank_correlation_uniqueness(train_loader_single_batch, non_train_loader_single_batch, model, loss, optimizer)
    gradient_correlation_unique = unsampled_indexes[most_decor_points]
    
    # Reset the model
    model = SimpleMLP(input_size=784, output_size=10, hidden_size=config.model.hidden_size, num_layers=config.model.num_layers).to(device)
    
    random_sampling_results = []
    uncertainty_sampling_results = []
    gradient_correlation_results = []
    gradient_correlation_cond_results = []
    gradient_correlation_unique_results = []
    for n_samples in n_additional_samples:
        # Random sampling
        random_sampling_data = indexed_data + torch.utils.data.Subset(train_data, passive_sampling_ranking[:n_samples])
        train_data_loader = torch.utils.data.DataLoader(random_sampling_data, batch_size=config.training.batch_size, shuffle=True, pin_memory=True, )
        current_model = deepcopy(model)
        current_optimizer = torch.optim.Adam(current_model.parameters(), lr=config.training.learning_rate)
        random_sampling_results_epoch = []
        validation_res = []
        for epoch in range(config.training.num_epochs):
            for j in range(10):
                train_epoch(current_model, current_optimizer, loss, train_data_loader, device)
            validation_res.append(validate(current_model, loss, val_loader, device))
        # add the validation score with highest accuracy
        best_epoch = max(validation_res, key=lambda x: x[1])
        random_sampling_results_epoch.append(best_epoch)
        random_sampling_results.append(list(zip(*random_sampling_results_epoch)))
        
        # Uncertainty sampling
        uncertainty_sampling_data = indexed_data + torch.utils.data.Subset(train_data, uncertainty_sampling_ranking[:n_samples])
        train_data_loader = torch.utils.data.DataLoader(uncertainty_sampling_data, batch_size=config.training.batch_size, shuffle=True, pin_memory=True, )
        current_model = deepcopy(model)
        current_optimizer = torch.optim.Adam(current_model.parameters(), lr=config.training.learning_rate)
        uncertainty_sampling_results_epoch = []
        validation_res = []
        for epoch in range(config.training.num_epochs):
            for j in range(10):
                train_epoch(current_model, current_optimizer, loss, train_data_loader, device)
            validation_res.append(validate(current_model, loss, val_loader, device))
        best_epoch = max(validation_res, key=lambda x: x[1])
        uncertainty_sampling_results_epoch.append(best_epoch)
        uncertainty_sampling_results.append(list(zip(*uncertainty_sampling_results_epoch)))
            
        # Gradient correlation sampling
        gradient_correlation_data = indexed_data + torch.utils.data.Subset(train_data, gradient_correlation_ranking[:n_samples])
        train_data_loader = torch.utils.data.DataLoader(gradient_correlation_data, batch_size=config.training.batch_size, shuffle=True, )
        current_model = deepcopy(model)
        current_optimizer = torch.optim.Adam(current_model.parameters(), lr=config.training.learning_rate)
        gradient_correlation_results_epoch = []
        validation_res = []
        for epoch in range(config.training.num_epochs):
            for j in range(10):
                train_epoch(current_model, current_optimizer, loss, train_data_loader, device)
            validation_res.append(validate(current_model, loss, val_loader, device))
        best_epoch = max(validation_res, key=lambda x: x[1])
        gradient_correlation_results_epoch.append(best_epoch)
        gradient_correlation_results.append(list(zip(*gradient_correlation_results_epoch)))
        
        # Gradient correlation sampling with conditioning on training data
        gradient_correlation_data = indexed_data + torch.utils.data.Subset(train_data, gradient_correlation_cond_ranking[:n_samples])
        train_data_loader = torch.utils.data.DataLoader(gradient_correlation_data, batch_size=config.training.batch_size, shuffle=True, )
        current_model = deepcopy(model)
        current_optimizer = torch.optim.Adam(current_model.parameters(), lr=config.training.learning_rate)
        gradient_correlation_cond_results_epoch = []
        validation_res = []
        for epoch in range(config.training.num_epochs):
            for j in range(10):
                train_epoch(current_model, current_optimizer, loss, train_data_loader, device)
            validation_res.append(validate(current_model, loss, val_loader, device))
        best_epoch = max(validation_res, key=lambda x: x[1])
        gradient_correlation_cond_results_epoch.append(best_epoch)
        gradient_correlation_cond_results.append(list(zip(*gradient_correlation_cond_results_epoch)))
        
        # Gradient correlation sampling with correlation to training data gradient
        gradient_correlation_data = indexed_data + torch.utils.data.Subset(train_data, gradient_correlation_unique[:n_samples])
        train_data_loader = torch.utils.data.DataLoader(gradient_correlation_data, batch_size=config.training.batch_size, shuffle=True, )
        current_model = deepcopy(model)
        current_optimizer = torch.optim.Adam(current_model.parameters(), lr=config.training.learning_rate)
        gradient_correlation_unique_results_epoch = []
        validation_res = []
        for epoch in range(config.training.num_epochs):
            for j in range(10):
                train_epoch(current_model, current_optimizer, loss, train_data_loader, device)
            validation_res.append(validate(current_model, loss, val_loader, device))
        best_epoch = max(validation_res, key=lambda x: x[1])
        gradient_correlation_unique_results_epoch.append(best_epoch)
        gradient_correlation_unique_results.append(list(zip(*gradient_correlation_unique_results_epoch)))
        
    return random_sampling_results, uncertainty_sampling_results, gradient_correlation_results, gradient_correlation_cond_results, gradient_correlation_unique_results

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
n_rep = 3
results = []
for i in tqdm(range(n_rep), desc='Running Repetitions'):
    indexes = torch.randperm(len(train_data))
    results.append(run_single_experiment(indexes[:n_start_data], indexes[n_start_data:], n_samples))
    
result_array = torch.tensor(results)
sample_validation_losses = result_array[:, 0, :, 0, :]
sample_validation_accuracy = result_array[:, 0, :, 1, :]
sample_validation_unc_losses = result_array[:, 1, :, 0, :]
sample_validation_unc_accuracy = result_array[:, 1, :, 1, :]
sample_validation_corr_losses = result_array[:, 2, :, 0, :]
sample_validation_corr_accuracy = result_array[:, 2, :, 1, :]
sample_validation_corr_cond_losses = result_array[:, 3, :, 0, :]
sample_validation_corr_cond_accuracy = result_array[:, 3, :, 1, :]
sample_validation_corr_unique_losses = result_array[:, 4, :, 0, :]
sample_validation_corr_unique_accuracy = result_array[:, 4, :, 1, :]


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


def generate_uncertainty_curve(ax, data_sample, label, color, significance_level=0.05):
    mean = data_sample.mean(0)
    std = data_sample.std(0)
    n_root = torch.sqrt(torch.tensor([data_sample.shape[0]]))
    q_val = scipy.stats.norm().ppf(1 - significance_level / 2)
    lines = q_val*std/n_root
    ax.plot(n_samples, mean, label=label, color=color)
    ax.fill_between(n_samples, mean - lines, mean + lines, color=color, alpha=0.2)

fig, axs = plt.subplots(2)
generate_uncertainty_curve(axs[0], sample_validation_losses[..., -1], 'Final Loss PL', 'red')
generate_uncertainty_curve(axs[0], sample_validation_unc_losses[..., -1], 'Final Loss unc', 'green')
generate_uncertainty_curve(axs[0], sample_validation_corr_losses[..., -1], 'Final Loss corr', 'blue')
generate_uncertainty_curve(axs[0], sample_validation_corr_cond_losses[..., -1], 'Final Loss corr cond', 'purple')
generate_uncertainty_curve(axs[0], sample_validation_corr_unique_losses[..., -1], 'Final Loss corr unique', 'orange')
axs[0].set_title('Final Loss')
axs[0].set_xlabel('Number of Samples')
axs[0].set_ylabel('Loss')
axs[0].set_xscale('log')
axs[0].legend()

generate_uncertainty_curve(axs[1], sample_validation_accuracy[..., -1], 'Final Accuracy PL', 'red')
generate_uncertainty_curve(axs[1], sample_validation_unc_accuracy[..., -1], 'Final Accuracy unc', 'green')
generate_uncertainty_curve(axs[1], sample_validation_corr_accuracy[..., -1], 'Final Accuracy corr', 'blue')
generate_uncertainty_curve(axs[1], sample_validation_corr_cond_accuracy[..., -1], 'Final Accuracy corr cond', 'purple')
generate_uncertainty_curve(axs[1], sample_validation_corr_unique_accuracy[..., -1], 'Final Accuracy corr unique', 'orange')

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