import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from typing import Optional
from dataclasses import dataclass

@dataclass
class LoRAConfig:
    r: int = 8
    alpha: int = 16
    dropout: float = 0.1
    merge_weights: bool = False

class LoRALinear(nn.Module):
    def __init__(
        self,
        in_features: int,
        out_features: int,
        config: LoRAConfig,
    ):
        super().__init__()
        self.config = config

        # Create regular weights and initialize them
        self.weight = nn.Parameter(torch.empty(out_features, in_features))
        self.bias = nn.Parameter(torch.zeros(out_features))
        
        # LoRA weights
        self.lora_A = nn.Parameter(torch.zeros(config.r, in_features))
        self.lora_B = nn.Parameter(torch.zeros(out_features, config.r))
        self.scaling = config.alpha / config.r
        self.dropout = nn.Dropout(p=config.dropout)
        
        # Initialize weights
        nn.init.kaiming_uniform_(self.weight, a=math.sqrt(5))
        nn.init.kaiming_uniform_(self.lora_A, a=math.sqrt(5))
        nn.init.zeros_(self.lora_B)
        
        # Freeze the regular weights if we're using LoRA
        self.weight.requires_grad = False
        self.bias.requires_grad = False
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Regular forward pass
        base_output = F.linear(x, self.weight, self.bias)
        
        # LoRA forward pass
        lora_output = self.dropout(x) @ self.lora_A.t() @ self.lora_B.t()
        
        return base_output + self.scaling * lora_output
        
    def merge_weights(self):
        """Merges LoRA weights into base weights for inference"""
        if self.config.merge_weights:
            self.weight.data += self.scaling * self.lora_B @ self.lora_A
            self.lora_A.data.zero_()
            self.lora_B.data.zero_()
            
    def unmerge_weights(self):
        """Unmerges LoRA weights from base weights"""
        if self.config.merge_weights:
            self.weight.data -= self.scaling * self.lora_B @ self.lora_A

def replace_linear_with_lora(model: nn.Module, config: LoRAConfig) -> None:
    """Recursively replaces linear layers with LoRA layers"""
    for name, module in model.named_children():
        if isinstance(module, nn.Linear):
            setattr(model, name, LoRALinear(
                module.in_features,
                module.out_features,
                config
            ))
        else:
            replace_linear_with_lora(module, config)
            
def get_lora_params(model: nn.Module):
    """Gets only the LoRA parameters for training"""
    params = []
    for module in model.modules():
        if isinstance(module, LoRALinear):
            params.extend([module.lora_A, module.lora_B])
    return params

def merge_lora_weights(model: nn.Module):
    """Merges LoRA weights into base weights for the whole model"""
    for module in model.modules():
        if isinstance(module, LoRALinear):
            module.merge_weights()
            
def unmerge_lora_weights(model: nn.Module):
    """Unmerges LoRA weights from base weights for the whole model"""
    for module in model.modules():
        if isinstance(module, LoRALinear):
            module.unmerge_weights()