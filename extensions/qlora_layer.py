# import torch
# import torch.nn as nn
# import bitsandbytes as bnb  # QLoRA requires bitsandbytes for 4-bit quantization
# import math

# class QLoraLayer(nn.Module):
#     def __init__(self, in_dim, out_dim, rank, alpha):
#         super(QLoraLayer, self).__init__()
#         self.A = nn.Parameter(torch.Tensor(in_dim, rank))
#         self.B = nn.Parameter(torch.Tensor(out_dim, rank))
#         self.alpha = alpha
#         nn.init.xavier_uniform_(self.A)
#         nn.init.kaiming_uniform_(self.B, a=math.sqrt(5))

#     def forward(self, x):
#         return self.alpha * x @ self.A @ self.B.t()

# class QuantizedLinearWithLora(nn.Module):
#     def __init__(self, in_dim, out_dim, rank, alpha):
#         super(QuantizedLinearWithLora, self).__init__()
        
#         # Apply 4-bit quantization to Linear layer
#         self.linear = bnb.nn.Linear4bit(in_dim, out_dim, bias=True)
        
#         # LoRA Adapter
#         self.lora = QLoraLayer(in_dim, out_dim, rank, alpha)

#     def forward(self, x):
#         return self.linear(x) + self.lora(x)

# def replace_linear_with_qlora(model, rank=8, alpha=16):
#     """
#     Replaces all `torch.nn.Linear` layers in the model with 4-bit quantized 
#     LoRA-enhanced layers for efficient fine-tuning.
    
#     Args:
#         model (torch.nn.Module): The base model to modify.
#         rank (int): The rank of the LoRA adaptation.
#         alpha (int): Scaling factor for LoRA.
    
#     Returns:
#         torch.nn.Module: The modified model with QLoRA applied.
#     """
#     last_layer_name, _ = list(model.named_modules())[-1]                  
    
#     for name, module in model.named_children():
#         if name != last_layer_name and isinstance(module, nn.Linear):
#             setattr(model, name, QuantizedLinearWithLora(module.in_features, module.out_features, rank, alpha))
#         else:
#             replace_linear_with_qlora(module, rank, alpha)

#     return model

# def freeze_all_except_lora(model, verbose=False):
#     """
#     Freezes all layers except LoRA layers.
    
#     Args:
#         model (torch.nn.Module): The model to modify.
#         verbose (bool): If True, prints which parameters remain trainable.
    
#     Returns:
#         None
#     """
#     for name, param in model.named_parameters():
#         if "lora" not in name:
#             param.requires_grad = False  # Freeze non-LoRA parameters
    
#     if verbose:
#         for name, param in model.named_parameters():
#             print(f"{name}: requires_grad = {param.requires_grad}")



import torch
import torch.nn as nn
import bitsandbytes as bnb
import math
def unfreeze_last(model, verbose=True):
    # Freeze all layers except the last layer in the model.
    # model: the PyTorch model
    last_layer_name, _ = list(model.named_modules())[-1]
    for name, module in model.named_children():
        if name == last_layer_name:
            for param in module.parameters():
                param.requires_grad = True
        else:
            unfreeze_last(module)
    if verbose:
        for name, param in model.named_parameters():
            print(f"{name}: requires_grad = {param.requires_grad}")


import torch
import torch.nn as nn
import bitsandbytes as bnb
import math


class QLoraLayer(nn.Module):
    def __init__(self, in_dim, out_dim, rank, alpha):
        super(QLoraLayer, self).__init__()
        self.in_dim = in_dim
        self.out_dim = out_dim
        self.rank = rank
        self.alpha = alpha
        
        # Initialize A with small random values and B with zeros
        self.A = nn.Parameter(torch.empty(in_dim, rank))
        self.B = nn.Parameter(torch.zeros(rank, out_dim))
        
        # Initialize A with small random values
        nn.init.normal_(self.A, std=math.sqrt(1.0 / in_dim))
        
        # Scale factor for forward pass
        self.scaling = alpha / rank
    
    def forward(self, x):
        # More numerically stable implementation with built-in scaling
        return (x @ self.A @ self.B) * self.scaling


class QuantizedLinearWithLora(nn.Module):
    def __init__(self, linear_module, rank, alpha, bias=True):
        super(QuantizedLinearWithLora, self).__init__()
        
        # Get dimensions directly from the original module
        self.in_features = linear_module.in_features
        self.out_features = linear_module.out_features
        
        # Create quantized linear layer
        self.linear = bnb.nn.Linear4bit(
            self.in_features, 
            self.out_features,
            bias=bias,
            compute_dtype=torch.float16  # Use fp16 for compute operations
        )
        
        # Copy weights if the linear_module has weights
        if hasattr(linear_module, 'weight'):
            # For Linear4bit, we need special handling
            if hasattr(self.linear, 'weight') and linear_module.weight.shape == self.linear.weight.shape:
                self.linear.weight.data = linear_module.weight.data
        
        # Copy bias if it exists - with dimension check
        if bias and hasattr(linear_module, 'bias') and linear_module.bias is not None:
            if linear_module.bias.shape == self.linear.bias.shape:
                self.linear.bias.data.copy_(linear_module.bias.data)
            else:
                print(f"Warning: Bias dimension mismatch - expected {self.linear.bias.shape}, got {linear_module.bias.shape}")
                # Initialize with zeros instead of copying
                nn.init.zeros_(self.linear.bias)
        
        # Add LoRA adapter
        self.lora = QLoraLayer(self.in_features, self.out_features, rank, alpha)
    
    def forward(self, x):
        # Combine the outputs
        return self.linear(x) + self.lora(x)


def replace_linear_with_qlora(model, rank=8, alpha=16, target_modules=None, exclude_modules=None):
    """
    Replace selected linear layers with quantized LoRA-enhanced versions.
    
    Args:
        model: The PyTorch model to modify
        rank: Rank for LoRA adapters (default: 8)
        alpha: Scaling factor for LoRA contribution (default: 16)
        target_modules: List of module names to apply QLoRA to (if None, apply to all linear layers)
        exclude_modules: List of module names to exclude from QLoRA application
        
    Returns:
        Modified model with QLoRA applied
    """
    # Convert to sets for faster lookup if provided
    if target_modules is not None:
        target_modules = set(target_modules)
    if exclude_modules is not None:
        exclude_modules = set(exclude_modules)
    
    # Recursively process all modules
    for name, module in list(model.named_children()):
        # Check if this module should be processed
        skip_module = (
            (exclude_modules is not None and name in exclude_modules) or
            (target_modules is not None and name not in target_modules)
        )
        
        if not skip_module and isinstance(module, nn.Linear):
            try:
                # Replace this linear module with a quantized LoRA version
                new_module = QuantizedLinearWithLora(
                    module,
                    rank=rank,
                    alpha=alpha,
                    bias=module.bias is not None
                )
                setattr(model, name, new_module)
                print(f"Successfully applied QLoRA to {name}")
            except Exception as e:
                print(f"Error applying QLoRA to {name}: {str(e)}")
                # Continue without replacing this module
                continue
        elif len(list(module.children())) > 0:
            # Recursively process child modules (only if it has children)
            child_targets = None
            if target_modules is not None:
                child_targets = [t.split('.', 1)[1] for t in target_modules 
                               if t.startswith(f"{name}.") and '.' in t]
                if not child_targets:
                    child_targets = None
                    
            child_excludes = None
            if exclude_modules is not None:
                child_excludes = [e.split('.', 1)[1] for e in exclude_modules 
                                if e.startswith(f"{name}.") and '.' in e]
                if not child_excludes:
                    child_excludes = None
            
            # Recursive call for children
            replace_linear_with_qlora(module, rank, alpha, child_targets, child_excludes)
    
    return model


def freeze_base_model_params(model, verbose=False):
    """
    Freeze all parameters except LoRA parameters.
    
    Args:
        model: The PyTorch model
        verbose: Whether to print parameter status
        
    Returns:
        Modified model with frozen parameters
    """
    # Set all parameters to non-trainable by default
    for name, param in model.named_parameters():
        param.requires_grad = False
    
    # Find and unfreeze all LoRA parameters
    trainable_params = 0
    total_params = 0
    
    for name, module in model.named_modules():
        # Unfreeze LoRA parameters
        if isinstance(module, QLoraLayer):
            for param_name, param in module.named_parameters():
                param.requires_grad = True
                trainable_params += param.numel()
                if verbose:
                    print(f"Trainable: {name}.{param_name}")
        
        # Count all parameters
        if isinstance(module, nn.Module):
            for param in module.parameters():
                total_params += param.numel()
    
    if verbose:
        print(f"Total parameters: {total_params:,}")
        print(f"Trainable parameters: {trainable_params:,} ({100 * trainable_params / total_params:.2f}%)")
    
    return model

