import torch
import torch.nn as nn
from typing import List, Dict


class Solution:

    def compute_activation_stats(self, model: nn.Module, x: torch.Tensor) -> List[Dict[str, float]]:
        # Forward pass through model layer by layer
        # After each nn.Linear, record: mean, std, dead_fraction
        # Run with torch.no_grad(). Round to 4 decimals.
        stats = []
        with torch.no_grad():
            for layer in model.children():
                x = layer(x)
                if isinstance(layer, nn.Linear):
                    mean = x.mean().item()
                    std = x.std().item()
                    is_zero_or_less = (x <= 0)
                    is_dead_neuron = is_zero_or_less.all(dim=0)
                    dead_frac = is_dead_neuron.float().mean().item()
                    stats.append({
                        "mean": round(mean, 4),
                        "std": round(std, 4),
                        "dead_fraction": round(dead_frac, 4)
                    })
        return stats

    def compute_gradient_stats(self, model: nn.Module, x: torch.Tensor, y: torch.Tensor) -> List[Dict[str, float]]:
        # Forward + backward pass with nn.MSELoss
        # For each nn.Linear layer's weight gradient, record: mean, std, norm
        # Call model.zero_grad() first. Round to 4 decimals.
        model.zero_grad()
        predictions = model(x)
        loss_fn = nn.MSELoss()
        loss = loss_fn(predictions, y)
        loss.backward()
        stats = []

        for layer in model.children():

            if isinstance(layer, nn.Linear):
                grad = layer.weight.grad
                mean = grad.mean().item()
                std = grad.std().item()
                norm = torch.norm(grad).item()
                stats.append({
                    "mean": round(mean, 4),
                    "std": round(std, 4),
                    "norm": round(norm, 4)
                })

        return stats

    def diagnose(self, activation_stats: List[Dict[str, float]], gradient_stats: List[Dict[str, float]]) -> str:
        # Classify network health based on the stats
        # Return: 'dead_neurons', 'exploding_gradients', 'vanishing_gradients', or 'healthy'
        # Check in priority order (see problem description for thresholds)
        if any(a["dead_fraction"] > 0.5 for a in activation_stats):
            return 'dead_neurons'
        
        if any(g["norm"] > 1000 for g in gradient_stats):
            return 'exploding_gradients'

        if any(g["norm"] < 1e-5 for g in gradient_stats):
            return 'vanishing_gradients'

        if any(a["std"] < 0.1 for a in activation_stats):
            return 'vanishing_gradients'

        if any(a["std"] > 10 for a in activation_stats):
            return 'exploding_gradients'

        return 'healthy'
