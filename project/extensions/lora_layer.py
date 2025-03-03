import math
import torch

class LoraLayer(torch.nn.Module):
    def __init__(self, in_dim, out_dim, rank, alpha):
        super(LoraLayer, self).__init__()
        self.A = torch.nn.Parameter(torch.Tensor(in_dim, rank))
        self.B = torch.nn.Parameter(torch.Tensor(out_dim, rank))
        self.alpha = alpha
        torch.nn.init.xavier_uniform_(self.A)

    def forward(self, x):
        # Compute the output of the LoraLayer.
        # x: [batch_size, in_dim]
        # output: [batch_size, out_dim]
        x =self.alpha * x @ self.A @ self.B.t()
        return x

class LinearWithLora(torch.nn.Module):
    def __init__(self, in_dim, out_dim, rank, alpha):
        super(LinearWithLora, self).__init__()
        self.linear = torch.nn.Linear(in_dim, out_dim)
        self.lora = LoraLayer(in_dim, out_dim, rank, alpha)

    def forward(self, x):
        # Compute the output of the LinearWithLora layer.
        # x: [batch_size, in_dim]
        # output: [batch_size, out_dim]
        return self.linear(x) + self.lora(x)
    
def replace_linear_with_lora(model, rank, alpha):
    # Replace all linear layers in the model with LinearWithLora layers.
    # model: the PyTorch model
    # rank: the rank of the LoraLayer
    # alpha: the alpha parameter of the LoraLayer
    last_layer_name, _ = list(model.named_modules())[-1]
    for name, module in model.named_children():
        if isinstance(module, torch.nn.Linear) and name not in last_layer_name:
            setattr(model, name, LinearWithLora(module.in_features, module.out_features, rank, alpha))

        else:
            replace_linear_with_lora(module, rank, alpha)

    return model

def freeze_all_but_last(model, verbose=False):
    # Freeze all layers except the last layer in the model.
    # model: the PyTorch model
    last_layer_name, _ = list(model.named_modules())[-1]
    for name, module in model.named_children():
        if name != last_layer_name:
            for param in module.parameters():
                param.requires_grad = False
        else:
            freeze_all_but_last(module)
    if verbose:
        for name, param in model.named_parameters():
            print(f"{name}: requires_grad = {param.requires_grad}")

