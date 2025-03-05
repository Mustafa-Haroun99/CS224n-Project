from datetime import datetime
import numpy as np
from tensorboard.backend.event_processing import event_accumulator

import os

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
        with open(file_name, 'w') as f:
            if not metrics_file_exists:
                columns = [f'{k}' for k in metrics.keys()]
                f.write(', '.join(columns))
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
    
    # Adjust these tag names based on what's in your TensorBoard file
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

# # Example usage
# log_dir = "path/to/your/logs/directory"  # This should be the directory containing the events file
# loss_array, acc_array = extract_tensorboard_data(log_dir)

# # Now you have two arrays:
# # loss_array with columns [step, loss_value]
# # acc_array with columns [step, accuracy_value]

# # Extract just the epoch numbers and values if needed
# epochs = loss_array[:, 0]
# loss_values = loss_array[:, 1]
# accuracy_values = acc_array[:, 1]

# print(f"Epochs: {epochs}")
# print(f"Loss values: {loss_values}")
# print(f"Accuracy values: {accuracy_values}")