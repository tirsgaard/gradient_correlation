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
        self.dropout = nn.Dropout(0.0)
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Flatten input
        x = x.view(-1, self.input_size)
        
        # Forward pass
        for layer in self.layers[:-1]:
            x = layer(x)
            x = self.dropout(x)
        x = self.layers[-1](x)
        return x