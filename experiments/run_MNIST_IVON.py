from typing import Iterable
import torch
import torch.nn.functional as F
import torchvision
import scipy
from src.models.NN_models import SimpleMLP, SingleLayerMLP
import yaml
from datasets.download_MNIST import get_MNIST
from src.training import train_epoch, validate, test, validate_IVON, train_IVON
from src.correlation_gradient import rank_sample_information, rank_correlation_uniqueness, rank_uncertainty_information
from tqdm import tqdm
import ivon
from src.utility import JS_div
from easydict import EasyDict as edict
import matplotlib.pyplot as plt
from copy import deepcopy

config = edict(yaml.safe_load(open('configs/MNIST_IVON.yaml', 'r')))
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

torch.manual_seed(1)
model = SimpleMLP(input_size=784, output_size=10, hidden_size=config.model.hidden_size, num_layers=config.model.num_layers).to(device)
loss = torch.nn.CrossEntropyLoss() #lambda x, y: 0.5*(x-y).pow(2).sum(-1).mean()  # 
loss_batched = torch.nn.CrossEntropyLoss(reduction='none')

# Load the data
train_data, val_data, test_data = get_MNIST(config.training.validation_split, device=device)
# Shuffle the train data
train_loader = torch.utils.data.DataLoader(train_data, batch_size=config.training.batch_size, shuffle=True, pin_memory=True)
val_loader = torch.utils.data.DataLoader(val_data, batch_size=config.training.batch_size, shuffle=True, pin_memory=True)
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
    model.eval()
    # Compute predictions of the non-indexed data
    y_hat = torch.cat([model(x.to(device)) for x, _ in data_loader])
    # Find most uncertain points
    most_unc_points = y_hat.softmax(-1).max(-1)[0].sort().indices
    return most_unc_points, y_hat.softmax(-1).max(-1)[0]

def uncertainty_IVON_rank_datapoints(data_loader: torch.utils.data.DataLoader, model: torch.nn.Module, optimizer: ivon.IVON, loss: torch.nn.Module, test_samples: int = 1) -> torch.Tensor:
    """ Rank the data points by uncertainty 
    Args:
        data_loader: the data loader to use
        model: the model to use
        optimizer: the optimizer to use
        loss: the loss function to use
        test_samples: the number of samples to use for testing
        
    Returns:
        most_unc_points: the indices of the most uncertain points
        unc: the uncertainty of the most certain class of all points
    """
    device = next(model.parameters()).device
    model.eval()
    all_props = []
    for x, y in data_loader:
        sampled_probs = []
        for i in range(test_samples):
            x, y = x.to(device), y.to(device)
            with optimizer.sampled_params():
                sampled_logit = model(x)
                sampled_probs.append(F.softmax(sampled_logit, dim=1).detach())
        prob = torch.mean(torch.stack(sampled_probs), dim=0)
        all_props.append(prob)
    prob = torch.cat(all_props, 0)
        
    # Find most uncertain points
    most_unc_points = prob.max(-1)[0].sort().indices
    unc = prob.max(-1)[0]
    return most_unc_points, unc

def QBC_rank_datapoints(data_loader: torch.utils.data.DataLoader, model: torch.nn.Module, optimizer: ivon.IVON, loss: torch.nn.Module, test_samples: int = 2) -> torch.Tensor:
    """ Rank the data points by querry by committee
    Args:
        data_loader: the data loader to use
        model: the model to use
        optimizer: the optimizer to use
        loss: the loss function to use
        test_samples: the number of samples to use for testing
        
    Returns:
        most_unc_points: the indices of the most uncertain points
    """
    device = next(model.parameters()).device
    model.eval()
    QBC_scores = []
    for x, y in data_loader:
        sampled_probs = []
        for i in range(test_samples):
            x, y = x.to(device), y.to(device)
            with optimizer.sampled_params():
                sampled_logit = model(x)
                sampled_probs.append(F.softmax(sampled_logit, dim=1))
        prob = torch.stack(sampled_probs)
        QBC_score = JS_div(prob.permute((1, 0, 2)))
        QBC_scores.append(QBC_score.detach())
    QBC_scores = torch.cat(QBC_scores, 0)
        
    # Find most uncertain points
    most_unc_points = QBC_scores.sort().indices
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
    optimizer = ivon.IVON(model.parameters(), lr=config.training.learning_rate, ess=len(sampled_indexes))

    # Split the data    
    reduced_unsample_indexes = unsampled_indexes[:max_add_subset]
    indexed_data = torch.utils.data.Subset(train_data, sampled_indexes)
    non_indexed_data = torch.utils.data.Subset(train_data, reduced_unsample_indexes)
    train_loader = torch.utils.data.DataLoader(indexed_data, batch_size=config.training.batch_size, shuffle=True, pin_memory=True)
    non_train_loader = torch.utils.data.DataLoader(non_indexed_data, batch_size=config.training.batch_size, shuffle=False, pin_memory=True)
    combined_loader = torch.utils.data.DataLoader(torch.utils.data.ConcatDataset([indexed_data, non_indexed_data]), batch_size=config.training.batch_size, shuffle=False, pin_memory=True)

    # Train the model
    validation_losses = []
    validation_accuracies = []
    best_model = None
    best_optimizer = None
    best_accuracy = 0
    
    for epoch in range(config.training.num_epochs):
        # Validate model
        val_loss, val_accuracy = validate_IVON(model, optimizer, loss, val_loader, device)
        validation_losses.append(val_loss)
        validation_accuracies.append(val_accuracy)
        # Update best model if necessary
        if best_model is None or val_accuracy > best_accuracy:
            best_model = model
            best_optimizer = optimizer
            best_accuracy = val_accuracy
        # Train
        for j in range(10):
            train_IVON(model, optimizer, loss, train_loader, device)
    print(f"Best Validation Accuracy: {best_accuracy} at epoch {validation_accuracies.index(best_accuracy)}")
    
    # Get rankings for different sampling strategies
    passive_sampling_ranking = reduced_unsample_indexes[torch.randperm(len(reduced_unsample_indexes))]
    
    # Uncertainty sampling
    most_unc_points, unc = uncertainty_IVON_rank_datapoints(non_train_loader, best_model, best_optimizer, loss)
    uncertainty_sampling_ranking = reduced_unsample_indexes[most_unc_points.cpu()]
    
    # IVON uncertainty sampling
    most_decor_points, _ = uncertainty_IVON_rank_datapoints(non_train_loader, best_model, best_optimizer, loss, test_samples=100)
    IVON_uncertainty_sampling_ranking = reduced_unsample_indexes[most_decor_points]
    
    # QBC sampling
    most_decor_points = QBC_rank_datapoints(non_train_loader, best_model, best_optimizer, loss, test_samples=10)
    QBC_ranking = reduced_unsample_indexes[most_decor_points]
    
    # Gradient correlation sampling with correlation to training data gradient
    most_decor_points, cov = rank_uncertainty_information(non_train_loader, best_model, loss_batched, best_optimizer, unc, pre_condition_index=torch.tensor([], dtype=int), cutoff_number=100)
    gradient_covariance_unique = reduced_unsample_indexes[most_decor_points]
    
    # Reset the model
    model = SimpleMLP(input_size=784, output_size=10, hidden_size=config.model.hidden_size, num_layers=config.model.num_layers).to(device)
    
    def run_sampling_experiment(data_rankings, n_samples, model, loss):
        data = indexed_data + torch.utils.data.Subset(train_data, data_rankings[:n_samples])
        train_data_loader = torch.utils.data.DataLoader(data, batch_size=config.training.batch_size, shuffle=True, pin_memory=True, ) 
        current_optimizer = ivon.IVON(model.parameters(), lr=config.training.learning_rate, ess=len(data))
        results_epoch = []
        validation_res = []
        for epoch in range(config.training.num_epochs):
            for j in range(10):
                train_IVON(model, current_optimizer, loss, train_data_loader, device)
            validation_res.append(validate_IVON(model, current_optimizer, loss, val_loader, device, 1))
        best_epoch = max(validation_res, key=lambda x: x[1])
        results_epoch.append(best_epoch)
        return list(zip(*results_epoch))
        
    random_sampling_results = []
    uncertainty_sampling_results = []
    IVON_uncertainty_results = []
    QBC_results = []
    gradient_covariance_unique_results = []
    for n_samples in n_additional_samples:
        # Random sampling
        random_sampling_results.append(run_sampling_experiment(passive_sampling_ranking, n_samples, deepcopy(model), loss))
        
        # Uncertainty sampling
        uncertainty_sampling_results.append(run_sampling_experiment(uncertainty_sampling_ranking, n_samples, deepcopy(model), loss))
            
        # Gradient correlation sampling
        IVON_uncertainty_results.append(run_sampling_experiment(IVON_uncertainty_sampling_ranking, n_samples, deepcopy(model), loss))
        
        # Gradient correlation sampling with conditioning on training data
        QBC_results.append(run_sampling_experiment(QBC_ranking, n_samples, deepcopy(model), loss))
        
        # Gradient correlation sampling with correlation to training data gradient
        gradient_covariance_unique_results.append(run_sampling_experiment(gradient_covariance_unique, n_samples, deepcopy(model), loss))
        
    return random_sampling_results, uncertainty_sampling_results, IVON_uncertainty_results, QBC_results, gradient_covariance_unique_results

n_start_data = 1000
n_rep = 10
results = []
for i in tqdm(range(n_rep), desc='Running Repetitions'):
    indexes = torch.randperm(len(train_data))
    results.append(run_single_experiment(indexes[:n_start_data], indexes[n_start_data:], n_samples, max_add_subset=10**3))
    
result_array = torch.tensor(results)
sample_validation_losses = result_array[:, 0, :, 0, :]
sample_validation_accuracy = result_array[:, 0, :, 1, :]
sample_validation_unc_losses = result_array[:, 1, :, 0, :]
sample_validation_unc_accuracy = result_array[:, 1, :, 1, :]
sample_validation_unc_IVON_losses = result_array[:, 2, :, 0, :]
sample_validation_unc_IVON_accuracy = result_array[:, 2, :, 1, :]
sample_validation_QBC_cond_losses = result_array[:, 3, :, 0, :]
sample_validation_QBC_cond_accuracy = result_array[:, 3, :, 1, :]
sample_validation_cov_losses = result_array[:, 4, :, 0, :]
sample_validation_cov_accuracy = result_array[:, 4, :, 1, :]

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
plt.savefig("figures/binary_MNIST_IVON_sample_runs.svg")
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
generate_uncertainty_curve(axs[0], sample_validation_unc_IVON_losses[..., -1], 'Final Loss IVON Unc', 'blue')
generate_uncertainty_curve(axs[0], sample_validation_QBC_cond_losses[..., -1], 'Final Loss corr cond', 'purple')
generate_uncertainty_curve(axs[0], sample_validation_cov_losses[..., -1], 'Final Loss cov', 'orange')
axs[0].set_title('Final Loss')
axs[0].set_xlabel('Number of Samples')
axs[0].set_ylabel('Loss')
axs[0].set_xscale('log')
axs[0].legend()

generate_uncertainty_curve(axs[1], sample_validation_accuracy[..., -1], 'Final Accuracy PL', 'red')
generate_uncertainty_curve(axs[1], sample_validation_unc_accuracy[..., -1], 'Final Accuracy unc', 'green')
generate_uncertainty_curve(axs[1], sample_validation_unc_IVON_accuracy[..., -1], 'Final Accuracy IVON Unc', 'blue')
generate_uncertainty_curve(axs[1], sample_validation_QBC_cond_accuracy[..., -1], 'Final Accuracy corr cond', 'purple')
generate_uncertainty_curve(axs[1], sample_validation_cov_accuracy[..., -1], 'Final Accuracy cov', 'orange')

axs[1].set_title('Final Accuracy')
axs[1].set_xlabel('Number of Samples')
axs[1].set_ylabel('Accuracy')
axs[1].set_xscale('log')
axs[1].legend()
plt.savefig("figures/MNIST_IVON_sample_runs_final.svg")
plt.show()


### Compare difference
fig, axs = plt.subplots()
generate_uncertainty_curve(axs, (sample_validation_unc_accuracy - sample_validation_accuracy)[..., -1], 'Uncertainty', 'green')
generate_uncertainty_curve(axs, (sample_validation_unc_IVON_accuracy - sample_validation_accuracy)[..., -1], 'Uncertainty IVON', 'blue')
generate_uncertainty_curve(axs, (sample_validation_QBC_cond_accuracy - sample_validation_accuracy)[..., -1], 'Correlation with Condition', 'purple')
generate_uncertainty_curve(axs, (sample_validation_cov_accuracy - sample_validation_accuracy)[..., -1], 'Covariance', 'orange')

axs.set_title('Final Accuracy')
axs.set_xlabel('Number of Samples')
axs.set_ylabel('Accuracy Difference from PL')
axs.set_xscale('log')
axs.legend()
plt.savefig("figures/MNIST_IVON_sample_runs_difference.svg")
plt.show()
