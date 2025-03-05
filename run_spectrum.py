
import torch
import numpy as np
# from prompt_toolkit.shortcuts import checkboxlist_dialog, input_dialog
import argparse
from tqdm import tqdm
import os

from extensions.spectrum import ModelModifier


def get_args():
    parser = argparse.ArgumentParser(description="Process SNR data for layers.")
    parser.add_argument("--seed", type=int, default=11711)
    parser.add_argument("--epochs", type=int, default=10)

    parser.add_argument("--batch_size", help='sst: 64, cfimdb: 8 can fit a 12GB GPU', type=int, default=8)
    parser.add_argument("--lr", type=float, help="learning rate", default=1e-5)
    parser.add_argument("--model_size", type=str,
                        help="The model size as specified on hugging face. DO NOT use the xl model.",
                        choices=['gpt2', 'gpt2-medium', 'gpt2-large'], default='gpt2')


    # Handle command-line arguments
   
    parser.add_argument('--model-name', type=str, required=True, help='Model name or path to the model')
    parser.add_argument('--top-percent', type=int, default=None, help='Top percentage of layers to select, overriding the default')
    args = parser.parse_args()

    return args

def main():
    args = get_args()
    # Check for existing SNR results file
    model_name_slug = args.model_name.replace('/', '-').replace('_', '-')
    snr_file_path = os.path.join('./extensions/spectrum','model_snr_results', f'snr_results_{model_name_slug}.json')
    device = torch.device('cuda') if  torch.cuda.is_available() else torch.device('cpu')
    from paraphrase_detection_ext import ParaphraseGPT, add_arguments
    args = add_arguments(args)
    model = ParaphraseGPT(args)
    model = model.to(device)
    batch_size = args.batch_size
    modifier = ModelModifier(model=model, model_name=args.model_name, batch_size=batch_size, top_percent=args.top_percent)
    selected_weight_types = modifier.interactive_select_weights()
    if selected_weight_types:
        modifier.assess_layers_snr(selected_weight_types)
        modifier.save_snr_to_json(snr_file_path)
        print("Finished SNR scanning and data saved.")
    else:
        print("No weight types selected.")

if __name__ == "__main__":
    main()







