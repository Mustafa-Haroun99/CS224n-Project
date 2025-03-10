# spectrum.py
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, AutoConfig
import numpy as np
import json
from prompt_toolkit.shortcuts import checkboxlist_dialog, input_dialog
import argparse
from tqdm import tqdm
import os
import time


import re


def unfreeze_last(model, verbose=False):
    "Function that unfreezes the last layer of a model"  
    last_layer_name, _ = list(model.named_modules())[-1]
    for name, module in model.named_children():
        print(name)
        if name == last_layer_name or name == 'gpt.word_embedding':
            for param in module.parameters():
                param.requires_grad = True
    if verbose:
        for name, param in model.named_parameters():
            print(f"{name}: requires_grad = {param.requires_grad}")


def freeze_model(model, file_name):
    with open(file_name, "r") as fin:
        yaml_parameters = fin.read()
   
    unfrozen_parameters = []
    for line in yaml_parameters.splitlines():
        if line.startswith("- "):
            unfrozen_parameters.append(line.split("- ")[1])

        def freeze_and_unfreeze_parameters(model, unfrozen_parameters):
            # freeze all parameters
            for param in model.parameters():
                param.requires_grad = False
            # unfreeze Spectrum parameters
            for name, param in model.named_parameters():
                if any(re.match(unfrozen_param, name) for unfrozen_param in unfrozen_parameters):
                    param.requires_grad = True

    freeze_and_unfreeze_parameters(model, unfrozen_parameters)


class ModelModifier:
    def __init__(self, model, model_name=None, top_percent=50, batch_size=1):
        self.model_name = model_name
        self.top_percent = top_percent
        self.batch_size = batch_size
        self.model = model
        self.layer_snr = {}
        self.layer_types = []

    def get_weight_types(self):
        weight_types = set()
        for name, module in self.model.named_modules():
            parts = name.split('.')
            if any(hasattr(module, attr) for attr in ['weight', 'bias','inv_freq']):
                layer_index = next((i for i, part in enumerate(parts) if part.isdigit()), -1)
                weight_type = '.'.join(parts[layer_index + 1:]) if layer_index != -1 else name
                weight_types.add(weight_type)
        return list(weight_types)

    def interactive_select_weights(self):
        weight_types = self.get_weight_types()
        sorted_weight_types = self.sort_weight_types(weight_types)
        selected_types = checkboxlist_dialog(
            title="Select Weight Types", 
            text="Deselect the weight types you do not want to scan for SNR:",
            values=[(wt, wt) for wt in sorted_weight_types],
            default_values=sorted_weight_types
        ).run()
        self.layer_types = selected_types
        return selected_types

    def sort_weight_types(self, weight_types):
        categories = {}
        for wt in weight_types:
            category = wt.split('.')[0]
            categories.setdefault(category, []).append(wt)
        sorted_categories = {k: sorted(v) for k, v in sorted(categories.items(), key=lambda item: item[0])}
        sorted_weight_types = [wt for sublist in sorted_categories.values() for wt in sublist]
        return sorted_weight_types

    def calculate_snr_for_layer(self, layer_type):
        layers = [(name, module) for name, module in self.model.named_modules() if layer_type in name and hasattr(module, 'weight')]
        num_batches = (len(layers) + self.batch_size - 1) // self.batch_size

        with tqdm(total=num_batches, unit='batch', desc=f'Calculating SNR for {layer_type}') as progress_bar:
            for i in range(0, len(layers), self.batch_size):
                batch_layers = layers[i:i + self.batch_size]
                for name, module in batch_layers:
                    weights = module.weight.detach()
                    if weights.ndim < 2:
                        weights = weights.unsqueeze(0)
                    S = torch.linalg.svdvals(weights)
                    max_singular_value = S[0]
                    sigma_estimated = self.estimate_sigma_with_full_iqr(S)
                    n, m = weights.shape[-2:]
                    mp_threshold = self.marchenko_pastur_threshold(sigma_estimated, n, m)
                    signal = S[S > mp_threshold].sum()
                    noise = S[S <= mp_threshold].sum()
                    snr = signal / noise if noise != 0 else float('inf')
                    snr_ratio = snr / max_singular_value
                    self.layer_snr[name] = {'type': layer_type, 'snr': snr_ratio.item()}
                progress_bar.update(1)

    @staticmethod
    def marchenko_pastur_threshold(sigma, n, m):
        beta = n / m if n < m else m / n
        threshold = sigma * np.sqrt((1 + np.sqrt(beta)) ** 2)
        return threshold

    @staticmethod
    def estimate_sigma_with_full_iqr(S):
        q75 = torch.quantile(S, 0.75)
        q25 = torch.quantile(S, 0.25)
        iqr = q75 - q25
        sigma_estimated = iqr / 1.349
        return sigma_estimated

    def assess_layers_snr(self, selected_weight_types):
        total_layers = sum(1 for name, module in self.model.named_modules() if any(layer_type in name for layer_type in selected_weight_types) and hasattr(module, 'weight'))
        start_time = time.time()

        with tqdm(total=len(selected_weight_types), unit='type', desc='Calculating SNR for types') as progress_bar:
            for layer_type in selected_weight_types:
                self.calculate_snr_for_layer(layer_type)
                progress_bar.update(1)

        end_time = time.time()
        total_time = end_time - start_time
        print(f"Total time taken: {total_time:.2f} seconds")

    def save_snr_to_json(self, directory=None):
        model_name_slug = self.model_name.replace('/', '-').replace('_', '-')
        directory = directory
        filename = os.path.join(directory, f'snr_results_{model_name_slug}.json')
        
        # Ensure the directory exists
        if not os.path.exists(directory):
            os.makedirs(directory)
        
        serializable_data = {}
        for layer_name, info in self.layer_snr.items():
            snr_value = info['snr'].item() if isinstance(info['snr'], torch.Tensor) else info['snr']
            layer_type = str(info['type'])
            serializable_data[layer_name] = {'snr': snr_value, 'type': layer_type}
        
        with open(filename, 'w') as file:
            json.dump(serializable_data, file, indent=4)
        
        print(f"Results saved to {filename}")
        self.save_top_snr_ratios_to_json(filename)
        self.generate_unfrozen_params_yaml(filename)

    def generate_unfrozen_params_yaml(self, json_filename, top_percent=None):
        top_percent = top_percent if top_percent is not None else self.top_percent
        with open(json_filename, 'r') as file:
            snr_data = json.load(file)
        unfrozen_parameters = {}
        for layer_name, info in snr_data.items():
            layer_type = info['type']
            if layer_type not in unfrozen_parameters:
                unfrozen_parameters[layer_type] = []
            unfrozen_parameters[layer_type].append((layer_name, info['snr']))
        top_layers_by_type = {}
        for layer_type, layers in unfrozen_parameters.items():
            layers_sorted = sorted(layers, key=lambda x: x[1], reverse=True)
            num_top_layers = int(len(layers) * top_percent / 100)
            top_layers_by_type[layer_type] = [layer[0] for layer in layers_sorted[:num_top_layers]]
        # Modify the yaml_filename to include the input json name and top_percent
        
        json_file_base = os.path.splitext(os.path.basename(json_filename))[0]
        yaml_filename = f"extensions/spectrum/model_snr_results/{json_file_base}_unfrozenparameters_{top_percent}percent.yaml"
        with open(yaml_filename, 'w') as file:
            file.write("unfrozen_parameters:\n")
            file.write("- ^lm_head.weight$\n")
            file.write("- ^model.embed_tokens.weight$\n")
            for layer_type, layer_names in top_layers_by_type.items():
                file.write(f"# {layer_type} layers\n")
                for layer_name in layer_names:
                    file.write(f"- {layer_name}\n")
        print(f"Top {top_percent}% SNR layers saved to {yaml_filename}")

    def save_top_snr_ratios_to_json(self, json_filename, filename=None):
        with open(json_filename, 'r') as file:
            snr_data = json.load(file)
        all_snr_layers = {}
        for layer_name, info in snr_data.items():
            layer_type = info['type']
            if layer_type not in all_snr_layers:
                all_snr_layers[layer_type] = []
            all_snr_layers[layer_type].append((layer_name, info['snr']))
        for layer_type, layers in all_snr_layers.items():
            layers_sorted = sorted(layers, key=lambda x: x[1], reverse=True)
            all_snr_layers[layer_type] = {layer[0]: layer[1] for layer in layers_sorted}

        json_file_base = os.path.splitext(os.path.basename(json_filename))[0]
        filename = f"{json_file_base}_sorted.json" if filename is None else filename

        with open(filename, 'w') as file:
            json.dump(all_snr_layers, file, indent=4)
        print(f"All SNR layers sorted and saved to {filename}")