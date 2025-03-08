import torch
import torch.nn as nn
import bitsandbytes as bnb  # QLoRA requires bitsandbytes for 4-bit quantization

class QLoraLayer(nn.Module):
    def __init__(self, in_dim, out_dim, rank, alpha):
        super(QLoraLayer, self).__init__()
        self.A = nn.Parameter(torch.Tensor(in_dim, rank))
        self.B = nn.Parameter(torch.Tensor(out_dim, rank))
        self.alpha = alpha
        nn.init.xavier_uniform_(self.A)

    def forward(self, x):
        return self.alpha * x @ self.A @ self.B.t()

class QuantizedLinearWithLora(nn.Module):
    def __init__(self, in_dim, out_dim, rank, alpha):
        super(QuantizedLinearWithLora, self).__init__()
        
        # Apply 4-bit quantization to Linear layer
        self.linear = bnb.nn.Linear4bit(in_dim, out_dim, bias=True)
        
        # LoRA Adapter
        self.lora = QLoraLayer(in_dim, out_dim, rank, alpha)

    def forward(self, x):
        return self.linear(x) + self.lora(x)

def replace_linear_with_qlora(model, rank=8, alpha=16):
    """
    Replaces all `torch.nn.Linear` layers in the model with 4-bit quantized 
    LoRA-enhanced layers for efficient fine-tuning.
    
    Args:
        model (torch.nn.Module): The base model to modify.
        rank (int): The rank of the LoRA adaptation.
        alpha (int): Scaling factor for LoRA.
    
    Returns:
        torch.nn.Module: The modified model with QLoRA applied.
    """
    for name, module in model.named_children():
        if isinstance(module, nn.Linear):
            setattr(model, name, QuantizedLinearWithLora(module.in_features, module.out_features, rank, alpha))
        else:
            replace_linear_with_qlora(module, rank, alpha)

    return model

def freeze_all_except_lora(model, verbose=False):
    """
    Freezes all layers except LoRA layers.
    
    Args:
        model (torch.nn.Module): The model to modify.
        verbose (bool): If True, prints which parameters remain trainable.
    
    Returns:
        None
    """
    for name, param in model.named_parameters():
        if "lora" not in name:
            param.requires_grad = False  # Freeze non-LoRA parameters
    
    if verbose:
        for name, param in model.named_parameters():
            print(f"{name}: requires_grad = {param.requires_grad}")

# Example usage
if __name__ == "__main__":
    # Create a simple model
    model = torch.nn.Sequential(
        torch.nn.Linear(768, 512),
        torch.nn.ReLU(),
        torch.nn.Linear(512, 256),
        torch.nn.ReLU(),
        torch.nn.Linear(256, 10)
    )
    
    model = replace_linear_with_qlora(model, rank=8, alpha=16)
    print(model)

    # Freeze all non-LoRA parameters
    freeze_all_except_lora(model, verbose=True)
