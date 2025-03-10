from datetime import datetime
import numpy as np
from tensorboard.backend.event_processing import event_accumulator
import glob
import os
import torch
import bitsandbytes as bnb

def generate_experiment_id():
    return datetime.now().strftime("%Y%m%d-%H%M%S")

def store_txt_experiment_data(metrics, 
                              task_name, directory_name= 'experiments'):
    directory_experiments  = os.path.join(directory_name, task_name)
    if not os.path.exists(directory_experiments):
        os.makedirs(directory_experiments)
    file_name = f'{directory_experiments}_metrics.txt'
    if metrics is not None:
        metrics_file_exists = os.path.exists(file_name)
        with open(file_name, 'a+') as f:
            if not metrics_file_exists:
                columns = [f'{k}' for k in metrics.keys()]
                f.write(', '.join(columns))
                f.write('\n')
            f.write('\n')
            row = [f'{v}' for v in metrics.values()]
            f.write(', '.join(row))
            f.write('\n')


def extract_tensorboard_data(log_dir):
    # Initialize an event accumulator
    ea = event_accumulator.EventAccumulator(log_dir,
        size_guidance={
            event_accumulator.SCALARS: 0,  # Get all scalar events
        })
    ea.Reload()  # Load the events
    
    # Get list of all scalar tags
    tags = ea.Tags()['scalars']
    print(f"Available tags: {tags}")
    
    # Extract loss and accuracy data
    loss_data = []
    acc_data = []
    
    # TODO: ADJUST THESE BASED ON WHAT WE HAVE IN OUR TENSORFLOW FILES
    loss_tag = "loss" if "loss" in tags else "train/loss"
    acc_tag = "accuracy" if "accuracy" in tags else "train/accuracy"
    
    if loss_tag in tags:
        loss_events = ea.Scalars(loss_tag)
        loss_data = [(event.step, event.value) for event in loss_events]
    
    if acc_tag in tags:
        acc_events = ea.Scalars(acc_tag)
        acc_data = [(event.step, event.value) for event in acc_events]
    
    # Convert to numpy arrays
    loss_array = np.array(loss_data)
    acc_array = np.array(acc_data)
    
    return loss_array, acc_array



def keep_latest_epoch_checkpoint(file_path, latest_epoch):
    # Get all checkpoint files
    checkpoint_files = glob.glob(f"{file_path}sonnet_*.pt")  # Adjust pattern if needed

    # Extract the epoch number from filenames
    def extract_epoch(file_name):
        return int(file_name.split("_")[-1].split(".")[0])  # Get number after last "_"

    # Iterate and delete files that are NOT the latest
    for file in checkpoint_files:
        epoch = extract_epoch(file)
        if epoch != latest_epoch:  # Keep only the specified latest epoch
            os.remove(file)
            print(f"Deleted: {file}")
    
    print(f"Kept latest checkpoint: {file_path}sonnet_{latest_epoch}.pt")

def print_requires_grad(model, parent_name=""):
    """
    Recursively prints whether each layer in the model has requires_grad=True.
    
    Args:
        model (torch.nn.Module): The PyTorch model.
        parent_name (str): Parent layer name (used for recursion).
    """
    for name, module in model.named_children():
        full_name = f"{parent_name}.{name}" if parent_name else name  # Construct full layer name
        
        # If the module has parameters, print whether requires_grad is True/False
        if any(p.requires_grad for p in module.parameters(recurse=False)):  
            status = "Trainable "
        else:
            status = "Frozen "
        
        print(f"Layer: {full_name} | {status}")

        # Recursively check submodules
        print_requires_grad(module, full_name)

def load_qlora_state_dict(model, state_dict):
    """
    Custom state_dict loader for QLoRA models that can handle both:
    1. Loading a standard model into a QLoRA-modified model
    2. Loading a QLoRA model into a QLoRA-modified model
    
    Args:
        model: The target model (with QLoRA applied)
        state_dict: The state dictionary to load
    
    Returns:
        The model with loaded weights
    """
    # Create a new state dict with only keys that exist in the model
    model_state_dict = model.state_dict()
    compatible_state_dict = {}
    
    # Filter quantization-specific parameters
    qlora_specific_keywords = [
        '.absmax', '.quant_map', '.nested_absmax', 
        '.nested_quant_map', '.quant_state', 'bitsandbytes'
    ]
    
    # Process each key in the loaded state dict
    for key, value in state_dict.items():
        # Check if this is a QLoRA-specific parameter
        is_qlora_param = any(keyword in key for keyword in qlora_specific_keywords)
        
        # If it's a standard parameter and exists in the model, keep it
        if not is_qlora_param and key in model_state_dict:
            compatible_state_dict[key] = value
        
        # If it's a QLoRA parameter and the corresponding key exists in the model, keep it
        elif is_qlora_param and key in model_state_dict:
            compatible_state_dict[key] = value
    
    # For debugging
    print(f"Loaded {len(compatible_state_dict)} parameters out of {len(model_state_dict)} available in the model")
    
    # Load the filtered state dict
    model.load_state_dict(compatible_state_dict, strict=False)
    return model


def dequantize_model_for_loading(model):
    """
    Convert a QLoRA model back to standard torch.nn.Linear layers
    to make it compatible with standard checkpoints.
    
    Args:
        model: The QLoRA model to convert
        
    Returns:
        The dequantized model
    """
    
    
    for name, module in list(model.named_children()):
        if isinstance(module, bnb.nn.Linear4bit):
    
            standard_linear = torch.nn.Linear(
                module.in_features, 
                module.out_features,
                bias=module.bias is not None
            )
            
        
            if hasattr(module, 'weight'):
                standard_linear.weight.data = module.weight.data.float()
            
            if module.bias is not None:
                standard_linear.bias.data = module.bias.data
                
            setattr(model, name, standard_linear)
        
        elif len(list(module.children())) > 0:
            dequantize_model_for_loading(module)
            
    return model