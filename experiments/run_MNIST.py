from typing import Iterable
import torch
import torchvision
import scipy
from src.models.NN_models import SimpleMLP, SingleLayerMLP, Parallel_MLP
import yaml
from datasets.download_MNIST import get_MNIST
from src.training import train_epoch, validate, test, validate_parallel
from src.correlation_gradient import rank_sample_information, rank_correlation_uniqueness, rank_uncertainty_information
from src.utility import JS_div, cross_entropy_parallel
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
cross_entropy_parallel = cross_entropy_parallel
loss_batched = torch.nn.CrossEntropyLoss(reduction='none')

# Load the data
train_data, val_data, test_data = get_MNIST(config.training.validation_split, device=device)
# Shuffle the train data
train_loader = torch.utils.data.DataLoader(train_data, batch_size=config.training.batch_size, shuffle=True, pin_memory=False)
val_loader = torch.utils.data.DataLoader(val_data, batch_size=config.training.batch_size, shuffle=True, pin_memory=False)
test_loader = torch.utils.data.DataLoader(test_data, batch_size=config.validation.batch_size, shuffle=True, pin_memory=False)

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
    model.eval()
    # Compute predictions of the non-indexed data
    y_hat = torch.cat([model(x.to(device)) for x, _ in data_loader])
    # Find most uncertain points
    most_unc_points = y_hat.softmax(-1).max(-1)[0].sort().indices
    return most_unc_points, y_hat.softmax(-1).max(-1)[0]
    
    
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
    reduced_unsample_indexes = unsampled_indexes[:max_add_subset]
    indexed_data = torch.utils.data.Subset(train_data, sampled_indexes)
    non_indexed_data = torch.utils.data.Subset(train_data, reduced_unsample_indexes)
    train_loader = torch.utils.data.DataLoader(indexed_data, batch_size=config.training.batch_size, shuffle=True, pin_memory=False)
    non_train_loader = torch.utils.data.DataLoader(non_indexed_data, batch_size=config.training.batch_size, shuffle=False, pin_memory=False)
    combined_loader = torch.utils.data.DataLoader(torch.utils.data.ConcatDataset([indexed_data, non_indexed_data]), batch_size=config.training.batch_size, shuffle=False, pin_memory=False)

    # Train the model
    validation_losses = []
    validation_accuracies = []
    best_model = None
    best_accuracy = 0
    
    for epoch in range(config.training.num_epochs):
        # Validate model
        val_loss, val_accuracy = validate(model, loss, val_loader, device)
        validation_losses.append(val_loss)
        validation_accuracies.append(val_accuracy)
        # Update best model if necessary
        if best_model is None or val_accuracy > best_accuracy:
            best_model = deepcopy(model)
            best_accuracy = val_accuracy
        
        # Train
        for j in range(10):
            train_epoch(model, optimizer, loss, train_loader, device)
    print(f"Best Validation Accuracy: {best_accuracy} at epoch {validation_accuracies.index(best_accuracy)}")
    
    # Get rankings for different sampling strategies
    passive_sampling_ranking = reduced_unsample_indexes[torch.randperm(len(reduced_unsample_indexes))]
    
    # Uncertainty sampling
    most_unc_points, unc = uncertainty_rank_datapoints(non_train_loader, best_model)
    uncertainty_sampling_ranking = reduced_unsample_indexes[most_unc_points.cpu()]
    
    # Gradient correlation sampling
    most_decor_points = rank_sample_information(non_train_loader, best_model, loss_batched, optimizer, pre_condition_index=torch.tensor([], dtype=int), cutoff_number=100)
    gradient_correlation_ranking = reduced_unsample_indexes[most_decor_points]
    
    # Gradient correlation sampling with conditioning on training data
    most_decor_points = rank_sample_information(combined_loader, best_model, loss_batched, optimizer, pre_condition_index=torch.tensor(range(len(indexed_data)), dtype=int), cutoff_number=100)
    gradient_correlation_cond_ranking = reduced_unsample_indexes[most_decor_points]
    
    # Gradient correlation sampling with correlation to training data gradient
    most_decor_points, cov = rank_uncertainty_information(non_train_loader, best_model, loss_batched, optimizer, unc, pre_condition_index=torch.tensor([], dtype=int), cutoff_number=100)
    gradient_covariance_unique = reduced_unsample_indexes[most_decor_points]
    
    rankings = [passive_sampling_ranking, uncertainty_sampling_ranking, gradient_correlation_ranking, gradient_correlation_cond_ranking, gradient_covariance_unique]
    
    # Reset the model
    model = Parallel_MLP(input_size=784, output_size=10, hidden_size=config.model.hidden_size, num_layers=config.model.num_layers, num_parallel=config.model.num_parallel).to(device)
    
    def run_sampling_experiment(data_rankings, n_samples, model, loss):
        data = indexed_data + torch.utils.data.Subset(train_data, data_rankings[:n_samples])
        train_data_loader = torch.utils.data.DataLoader(data, batch_size=config.training.batch_size, shuffle=True, pin_memory=False) 
        current_optimizer = torch.optim.Adam(model.parameters(), lr=config.training.learning_rate)
        results_epoch = []
        validation_res = []
        for epoch in range(config.training.num_epochs):
            for j in range(10):
                train_epoch(model, current_optimizer, cross_entropy_parallel, train_data_loader, device)
            validation_res.append(validate_parallel(model, cross_entropy_parallel, val_loader, device))
        best_epoch = max(validation_res, key=lambda x: x[1])
        results_epoch.append(best_epoch)
        return list(zip(*results_epoch))
        
    results = []
    for n_samples in n_additional_samples:
        results.append([run_sampling_experiment(data_rankings, n_samples, deepcopy(model), loss) for data_rankings in rankings])
    return results

n_start_data = 1000
n_rep = 2
run_experiments = True

if run_experiments:
    results = []
    for i in tqdm(range(n_rep), desc='Running Repetitions'):
        indexes = torch.randperm(len(train_data))
        results.append(run_single_experiment(indexes[:n_start_data], indexes[n_start_data:], n_samples, max_add_subset=10**3))
        
    result_array = torch.tensor(results)  # (n_rep, n_samples, n_sampling_strategies, 2, 2)
    torch.save(result_array, 'MNIST_results.pt')
else:
    result_array = torch.load('MNIST_results.pt')

sample_validation_losses = result_array[:, :, 0,  0, :]
sample_validation_accuracy = result_array[:, :, 0, 1, :]
sample_validation_unc_losses = result_array[:, :, 1, 0, :]
sample_validation_unc_accuracy = result_array[:, :, 1, 1, :]
sample_validation_corr_losses = result_array[:, :, 2, 0, :]
sample_validation_corr_accuracy = result_array[:, :, 2, 1, :]
sample_validation_corr_cond_losses = result_array[:, :, 3, 0, :]
sample_validation_corr_cond_accuracy = result_array[:, :, 3, 1, :]
sample_validation_cov_losses = result_array[:, :, 4, 0, :]
sample_validation_cov_accuracy = result_array[:, :, 4, 1, :]

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
generate_uncertainty_curve(axs[0], sample_validation_losses[..., -1], 'PL', 'red')
generate_uncertainty_curve(axs[0], sample_validation_unc_losses[..., -1], 'unc', 'green')
generate_uncertainty_curve(axs[0], sample_validation_corr_losses[..., -1], 'corr', 'blue')
generate_uncertainty_curve(axs[0], sample_validation_corr_cond_losses[..., -1], 'corr cond', 'purple')
generate_uncertainty_curve(axs[0], sample_validation_cov_losses[..., -1], 'cov', 'orange')
axs[0].set_title('Final Loss')
axs[0].set_xlabel('Number of Samples')
axs[0].set_ylabel('Loss')
axs[0].set_xscale('log')
axs[0].legend()

generate_uncertainty_curve(axs[1], sample_validation_accuracy[..., -1], 'Accuracy PL', 'red')
generate_uncertainty_curve(axs[1], sample_validation_unc_accuracy[..., -1], 'Accuracy unc', 'green')
generate_uncertainty_curve(axs[1], sample_validation_corr_accuracy[..., -1], 'Accuracy corr', 'blue')
generate_uncertainty_curve(axs[1], sample_validation_corr_cond_accuracy[..., -1], 'Accuracy corr cond', 'purple')
generate_uncertainty_curve(axs[1], sample_validation_cov_accuracy[..., -1], 'Accuracy cov', 'orange')

axs[1].set_title('Final Accuracy')
axs[1].set_xlabel('Number of Samples')
axs[1].set_ylabel('Accuracy')
axs[1].set_xscale('log')
axs[1].legend()
plt.savefig("figures/MNIST_sample_runs_final.svg")
plt.show()


### Compare difference
fig, axs = plt.subplots()
generate_uncertainty_curve(axs, (sample_validation_unc_accuracy - sample_validation_accuracy)[..., -1], 'Uncertainty', 'green')
generate_uncertainty_curve(axs, (sample_validation_corr_accuracy - sample_validation_accuracy)[..., -1], 'Correlation', 'blue')
generate_uncertainty_curve(axs, (sample_validation_corr_cond_accuracy - sample_validation_accuracy)[..., -1], 'Correlation with Condition', 'purple')
generate_uncertainty_curve(axs, (sample_validation_cov_accuracy - sample_validation_accuracy)[..., -1], 'Covariance', 'orange')

axs.set_title('Final Accuracy')
axs.set_xlabel('Number of Samples')
axs.set_ylabel('Accuracy Difference from PL')
axs.set_xscale('log')
axs.legend()
plt.savefig("figures/MNIST_sample_runs_difference.svg")
plt.show()