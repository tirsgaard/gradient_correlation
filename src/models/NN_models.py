import torch
import torchvision
import torch.nn as nn

class linear_batch_norm_relu(nn.Module):
    def __init__(self, in_features, out_features, bias=True):
        super(linear_batch_norm_relu, self).__init__()
        self.linear = nn.Linear(in_features, out_features, bias=bias)
        self.batch_norm = nn.BatchNorm1d(out_features)
        self.relu = nn.ReLU()
        
    def forward(self, x):
        x = self.linear(x)
        x = self.relu(x)
        x = self.batch_norm(x)
        return x
    
class linear_layer_norm_relu(nn.Module):
    def __init__(self, in_features, out_features, bias=True):
        super(linear_layer_norm_relu, self).__init__()
        self.linear = nn.Linear(in_features, out_features, bias=bias)
        self.layer_norm = nn.LayerNorm(out_features)
        self.relu = nn.ReLU()
        
    def forward(self, x):
        x = self.linear(x)
        x = self.relu(x)
        x = self.layer_norm(x)
        return x
    
class linear_relu(nn.Module):
    def __init__(self, in_features, out_features, bias=True):
        super(linear_relu, self).__init__()
        self.linear = nn.Linear(in_features, out_features, bias=bias)
        self.relu = nn.ReLU()
        
    def forward(self, x):
        x = self.linear(x)
        x = self.relu(x)
        return x
    

class SimpleMLP(nn.Module):
    """ A simple multi-layer perceptron for MNIST classification """
    
    def __init__(self, input_size: int, output_size: int, hidden_size: int, num_layers: int = 4):
        super(SimpleMLP, self).__init__()
        self.input_size = input_size
        self.output_size = output_size
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        
        # Define layers
        self.layers = nn.ModuleList()
        self.layers.append(nn.Linear(input_size, hidden_size))
        for _ in range(num_layers - 2):
            self.layers.append(linear_relu(hidden_size, hidden_size))
        self.layers.append(nn.Linear(hidden_size, output_size))
        self.dropout = nn.Dropout(0.)
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Flatten input
        x = x.view(-1, self.input_size)
        
        # Forward pass
        for layer in self.layers[:-1]:
            x = layer(x)
            x = self.dropout(x)
        x = self.layers[-1](x)
        return x
    
    
class own_linear_layer(nn.Module):
    def __init__(self, in_features, out_features, bias=True):
        super(own_linear_layer, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.weight = nn.Parameter(torch.Tensor(out_features, in_features))
        self.beta = 0.1
        if bias:
            self.bias = nn.Parameter(torch.Tensor(out_features))
        else:
            self.register_parameter('bias', None)
        self.reset_parameters()
        
    def reset_parameters(self):
        # Initialize the weights to normal distribution
        nn.init.normal_(self.weight, mean=0.0, std=1.0)
        if self.bias is not None:
            nn.init.normal_(self.bias, mean=0.0, std=1.0)
    
    def forward(self, x):
        return x @ self.weight.t()/(self.weight.shape[-1]**0.5) + self.beta*self.bias


class SingleLayerMLP(nn.Module):
    """ A simple single hidden-layer perceptron for MNIST classification """
    def __init__(self, input_size: int, output_size: int, hidden_size: int):
        super(SingleLayerMLP, self).__init__()
        self.input_size = input_size
        self.output_size = output_size
        self.hidden_size = hidden_size
        
        # Define layers
        self.layers = nn.ModuleList()
        self.layers.append(own_linear_layer(input_size, hidden_size))
        self.layers.append(nn.ReLU())
        self.layers.append(own_linear_layer(hidden_size, output_size))
        
        # Initialize weights
        for layer in self.layers:
            if isinstance(layer, own_linear_layer):
                layer.reset_parameters()
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Flatten input
        x = x.view(-1, self.input_size)
        
        # Forward pass
        for layer in self.layers:
            x = layer(x)
        return x
    
    def get_weights(self):
        return [layer.weight for layer in self.layers if isinstance(layer, own_linear_layer)]
    
    def get_biases(self):
        return [layer.bias for layer in self.layers if isinstance(layer, own_linear_layer)]
    
    def set_weights(self, weights, biases, initial_gain):
        i = 0
        for layer in self.layers:
            if isinstance(layer, own_linear_layer):
                weight = weights[i]
                bias = biases[i]
                in_features = layer.in_features
                out_features = layer.out_features
                k_factor = initial_gain #in_features/weight.shape[1])**0.5
                layer.weight.data = weight.data[:out_features, :in_features]*k_factor
                layer.bias.data = bias.data[:out_features]*k_factor
                i += 1
    
    
class CNN(nn.Module):
    """ A simple convolutional neural network for CIFAR100 classification """
    def __init__(self, input_shape: tuple[int, int, int], output_size: int, hidden_size: int, num_layers: int = 4):
        super(CNN, self).__init__()
        self.input_shape = input_shape
        self.output_size = output_size
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        
        self.layers = nn.ModuleList()
        self.layers.append(nn.Conv2d(input_shape[0], hidden_size, kernel_size=3, padding=1))
        self.layers.append(nn.ReLU())
        self.layers.append(nn.MaxPool2d(kernel_size=2, stride=2))
        for _ in range(num_layers - 2):
            self.layers.append(nn.Conv2d(hidden_size, hidden_size, kernel_size=3, padding=1))
            self.layers.append(nn.ReLU())
            self.layers.append(nn.MaxPool2d(kernel_size=2, stride=2))
        self.layers.append(nn.Flatten())
        self.layers.append(nn.Linear(hidden_size * (input_shape[1] // 2 ** (num_layers - 1)) * (input_shape[2] // 2 ** (num_layers - 1)), output_size))
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        for layer in self.layers:
            x = layer(x)
        return x
    
    
class linear_relu_parallel(nn.Module):
    def __init__(self, in_features: int, out_features: int, num_parallel: int, bias: bool = True):
        super(linear_relu_parallel, self).__init__()
        self.w1 = torch.nn.Parameter(torch.randn(num_parallel, in_features, out_features))
        self.w1 = torch.nn.init.kaiming_uniform_(self.w1)
        self.b1 = torch.nn.Parameter(torch.randn(num_parallel, out_features))
        self.b1.data = torch.nn.init.uniform_(self.b1.data, -1, 1)
        self.relu = nn.ReLU()
        
    def forward(self, x):
        x = x[:, None, :]
        x = self.relu(torch.einsum('ijk,nij->nik', self.w1, x) + self.b1)
        return x  # Shape (batch_size, num_parallel, out_features)
    
class Parallel_MLP(nn.Module):
    def __init__(self, input_size: int, output_size: int, hidden_size: int, num_layers: int = 4, num_parallel: int = 4):
        """ A parallel multi-layer perceptron.
        Args:
            input_size: The input size of the model
            output_size: The output size of the model
            hidden_size: The hidden size of the model
            num_layers: The number of layers in the model
            num_parallel: The number of parallel models to contain. This ultimately results in a ensemble model.
        """
        super(Parallel_MLP, self).__init__()
        self.input_size = input_size
        self.output_size = output_size
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.num_parallel = num_parallel
        self.layers = nn.ModuleList()
        self.layers.append(linear_relu_parallel(input_size, hidden_size, num_parallel))
        for _ in range(num_layers - 2):
            self.layers.append(linear_relu(hidden_size, hidden_size, num_parallel))
        self.layers.append(nn.Linear(hidden_size, output_size, num_parallel))
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x.view(-1, self.input_size)
        for layer in self.layers[:-1]:
            x = layer(x)
        x = self.layers[-1](x)
        return x
        
        