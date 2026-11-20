import torch


def neuron_forward(
    inputs: torch.Tensor, weights: torch.Tensor, bias: torch.Tensor
) -> tuple[torch.Tensor, torch.Tensor]:
  """Returns a tuple of 0-D scalar tensors: (preactivation, tanh output)."""
  weighted_sum = torch.sum(inputs * weights)
  preactivation = weighted_sum + bias
  output = torch.tanh(preactivation)
  return preactivation, output