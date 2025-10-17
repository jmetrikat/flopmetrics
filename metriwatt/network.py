"""Multi-layer perceptron network implementation."""
import torch
import torch.nn as nn

class SimpleLinear(nn.Module):
    """
    Simple linear layer with sigmoid activation function.
    """
    def __init__(
        self,
        in_features: int,
        out_features: int,
        device: torch.device,
        dtype: torch.dtype = torch.float32,
        *args,
        **kwargs,
    ) -> None:
        super().__init__(*args, **kwargs)

        self.in_features = in_features
        self.out_features = out_features

        self.W = nn.Parameter(
            torch.randn(in_features, out_features, dtype=dtype, device=device),
            requires_grad=True,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        Z = self.W @ x
        H = nn.functional.sigmoid(Z)
        return H

class ToyNetwork(nn.Module):
    """
    MLP network consisting of multiple linear layers.
    """
    def __init__(self, n_layers=10, dim=100, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.layers = nn.ModuleList([SimpleLinear(dim, dim, device=torch.device('cuda')) for _ in range(n_layers)])

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        for layer in self.layers:
            x = layer(x)
        return x

def run_toy_network_forward_ncu(dim, n_layers, n_tokens):
    net = ToyNetwork(n_layers=n_layers, dim=dim).to('cuda')
    x = torch.randn(dim, n_tokens, device='cuda')
    _ = net(x)

def run_toy_network_forward_backward_ncu(dim, n_layers, n_tokens):
    net = ToyNetwork(n_layers=n_layers, dim=dim).to('cuda')
    x = torch.randn(dim, n_tokens, device='cuda')
    y = net(x)
    y.sum().backward()

def construct_toy_network_and_input_for_ncu(dim, n_layers, n_tokens):
    _ = ToyNetwork(n_layers=n_layers, dim=dim).to('cuda')
    _ = torch.randn(dim, n_tokens, device='cuda')
