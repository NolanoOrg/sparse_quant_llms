"""Iteratively Prune a model based on the magnitude of weights.

Pytorch only supports x86/ARM for quantization.


"""
import collections # pylint: disable=syntax-error
import copy

from typing import Union, List # pylint: disable=syntax-error
import torch
from torch.nn.utils import prune as torch_prune # pylint: disable=wrong-import-position

def find_layers(module, layers=[nn.Conv2d, nn.Linear], name=''): # pylint: disable=dangerous-default-value
    """Find linear and conv layers in a model."""
    if type(module) in layers:
        return {name: module}
    res = {}
    for name1, child in module.named_children():
        res.update(find_layers(
            child, layers=layers, name=name + '.' + name1 if name != '' else name1
        ))
    return res

def log_prune_statistics(parameters_to_prune):
    """Logs the prune statistics."""
    total_num_pruned = 0
    total_num_params = 0
    for param_layer, _ in parameters_to_prune:
        this_layer_pruned = float(torch.sum(param_layer.weight == 0)) # pylint: disable=no-member
        this_num_params = float(param_layer.weight.nelement())
        total_num_pruned += this_layer_pruned
        total_num_params += this_num_params

        sparsity_percent = round(100. * this_layer_pruned / this_num_params, 3)
        print(f"Sparsity in {param_layer}: {sparsity_percent}%")

    print(f"Global sparsity: {round(100. * total_num_pruned / total_num_params, 3)}%")

class PruneInitialize:
    def __init__(self, model) -> None:
        self.model = model
        self.initialize()

    def initialize(self):
        """For each prunable layer we mask out all the weights that are zero."""
        layers = self.model.model.decoder.layers

        for i, layer in enumerate(layers):
            prunable_layers = find_layers(layer)
            for prunable_layer_name, prunable_layer in prunable_layers.items():
                # Find percentage of weights that are zero
                num_zeros = torch.sum(prunable_layer.weight == 0) # pylint: disable=no-member
                total_num_weights = prunable_layer.weight.nelement()
                percentage_zeros = num_zeros / total_num_weights
                print(f"Percentage of zeros in {prunable_layer_name}: {percentage_zeros}")

                torch_prune.random_unstructured(prunable_layer, name="weight", amount=percentage_zeros)
        
    def remove_prune(self):
        layers = self.model.model.decoder.layers
        for i, layer in enumerate(layers):
            prunable_layers = find_layers(layer)
            for prunable_layer_name, prunable_layer in prunable_layers.items():
                torch_prune.remove(prunable_layer, 'weight')

