'''
Sonnet generation starter code.

Running:
    `python sonnet_generation.py --use_gpu`

trains your SonnetGPT model and writes the required submission files.
'''
import os
import argparse
import random

import numpy as np
import torch.nn.functional as F
import torch
from torch import nn
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.utils.data import DataLoader
from tqdm import tqdm
from transformers import GPT2Tokenizer
from einops import rearrange
from torch.utils.tensorboard import SummaryWriter

from datasets import (
    SonnetsDataset,
)
from models.gpt2 import GPT2Model

# Local Imports
from models.gpt2 import GPT2Model
from extensions.lora_layer import replace_linear_with_lora, freeze_all_but_last
from extensions.qlora_layer import replace_linear_with_qlora, unfreeze_last
from extensions.spectrum import freeze_model, unfreeze_last
from extensions.jacobian_reg import JacobianReg
from extensions.pipeline_utils import store_txt_experiment_data, generate_experiment_id, keep_latest_epoch_checkpoint, print_requires_grad
from extensions.smart_pytorch import SMARTLoss
from extensions.smart_loss import kl_loss,sym_kl_loss
from extensions.early_stopper import EarlyStopping
from extensions.dropout_modifier import modify_model_dropout
from extensions.pipeline_utils import load_qlora_state_dict

from optimizer import AdamW

TQDM_DISABLE = False


# Fix the random seed.
def seed_everything(seed=11711):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True


class SonnetGPT(nn.Module):
    """Your GPT-2 Model designed for paraphrase detection."""

    def __init__(self, args):
        super().__init__()
        self.gpt = GPT2Model.from_pretrained(model=args.model_size, d=args.d, l=args.l, num_heads=args.num_heads)
        self.tokenizer = GPT2Tokenizer.from_pretrained('gpt2')
        self.tokenizer.pad_token = self.tokenizer.eos_token
        self.linear = nn.Linear(args.d, self.tokenizer.vocab_size)

        # By default, fine-tune the full model. TODO: this is maybe not idea.
        for param in self.gpt.parameters():
            param.requires_grad = True

    def forward(self, input_ids, attention_mask, return_input_embeddings=False):
        """
        This is similar to the forward for ParaphraseGPT, but we now want to produce a logit for each token in our sequence;
        not just the last token! This will allow our model to learn the natural language distribution that composes sonnets,
        not just the distribution over next tokens for the last token!
        """
        ### YOUR CODE HERE
        outputs = self.gpt(input_ids, attention_mask=attention_mask, return_embeddings=return_input_embeddings)
        # return logits
        output = self.gpt.hidden_state_to_token(outputs['last_hidden_state'])
        if return_input_embeddings:
            return output, outputs['embeddings']
        return output
    
    def get_embeddings(self, input_ids):
        return self.gpt.embed(input_ids)

    def forward_with_embeddings(self, embedding_output, attention_mask):
        outputs = self.gpt.run_transformer(embedding_output, attention_mask)
        output = self.gpt.hidden_state_to_token(outputs['last_hidden_state'])
        return output 


    def get_device(self):
        for param in self.gpt.parameters():
            return param.device

    @torch.no_grad()
    def generate(self, encoding, temperature=0.7, top_p=0.9, max_length=128, debug=False):
        """
        Generates an original sonnet using top-p sampling and softmax temperature.

        TODO: this is probably not ideal. You can look at hugging face's model.generate(...) function for inspiration.
        In particular, generating multiple sequences and choosing the best with beam search is one avenue. Top_k is another;
        there are many.
        """
        token_ids = encoding.to(self.get_device())
        attention_mask = torch.ones(token_ids.shape, dtype=torch.int64).to(self.get_device())
        I = 0

        for _ in range(max_length):
            if debug:
                if I == 10:
                    break
                I += 1
            # Forward pass to get logits
            logits_sequence = self.forward(token_ids, attention_mask)
            logits_last_token = logits_sequence[:, -1, :] / temperature  # Apply temperature scaling

            # Convert logits to probabilities
            probs = torch.nn.functional.softmax(logits_last_token, dim=-1)

            # Top-p (nucleus) sampling
            sorted_probs, sorted_indices = torch.sort(probs, descending=True)
            cumulative_probs = torch.cumsum(sorted_probs, dim=-1)
            top_p_mask = cumulative_probs <= top_p
            top_p_mask[..., 1:] = top_p_mask[..., :-1].clone()  # Shift mask right for proper thresholding
            top_p_mask[..., 0] = True  # Always include the highest probability token
            filtered_probs = sorted_probs * top_p_mask  # Zero out unlikely tokens
            filtered_probs /= filtered_probs.sum(dim=-1, keepdim=True)  # Normalize probabilities

            # Sample from filtered distribution
            sampled_index = torch.multinomial(filtered_probs, 1)
            sampled_token = sorted_indices.gather(dim=-1, index=sampled_index)

            # Stop if end-of-sequence token is reached
            if sampled_token.item() == self.tokenizer.eos_token_id:
                break

            # Append sampled token
            token_ids = torch.cat([token_ids, sampled_token], dim=1)
            attention_mask = torch.cat(
                [attention_mask, torch.ones((1, 1), dtype=torch.int64).to(self.get_device())], dim=1
            )

        generated_output = self.tokenizer.decode(token_ids[0].cpu().numpy().tolist())[3:]
        return token_ids, generated_output


def save_model(model, optimizer, args, filepath):
    save_info = {
        'model': model.state_dict(),
        'optim': optimizer.state_dict(),
        'args': args,
        'system_rng': random.getstate(),
        'numpy_rng': np.random.get_state(),
        'torch_rng': torch.random.get_rng_state(),
    }
    print(f"Saving model to {filepath}")
    torch.save(save_info, filepath)
    print(f"Save the model to {filepath}")


def train(args, experiment_id=1):
    """Train GPT-2 for paraphrase detection on the Quora dataset."""
    device = torch.device('cuda') if args.use_gpu else torch.device('cpu')
    if args.debug:
      print("\033[91m############### Debugging mode  is ON#############\033[0m")
      DEBUGGING = args.debug
    #### Instantiating Tensorboard Writter
    experiment_path = args.filepath.replace('.pt', '').replace('experiments/', 'runs/')
    save_model_dir = args.filepath.replace('.pt', f'')
    print(f"Saving model to {save_model_dir}")
    os.makedirs(save_model_dir, exist_ok=True)

    writer = SummaryWriter(f'{experiment_path}_{experiment_id}') # TODO: FIX PATH 

    # Create the data and its corresponding datasets and dataloader.
    sonnet_dataset = SonnetsDataset(args.sonnet_path)
    sonnet_dataloader = DataLoader(sonnet_dataset, shuffle=True, batch_size=args.batch_size,
                                                                 collate_fn=sonnet_dataset.collate_fn)

    # Create the held-out dataset: these only have the first 3 lines. Your job is to fill in the rest!
    held_out_sonnet_dataset = SonnetsDataset(args.held_out_sonnet_path)

    args = add_arguments(args)

    model = SonnetGPT(args)

    # Adding change in dropout rates
    if args.change_dropout:
        modify_model_dropout(model, args.dropout, args.attn_dropout)
   
   # Early stopping
    early_stopping = EarlyStopping(patience=args.patience, delta=args.delta)

    model = model.to(device)
    # Applying Spectrum
    if args.spectrum:
        weights_path = f"extensions/spectrum/model_snr_results/snr_results_gpt2_unfrozenparameters_{args.top_percent}percent.yaml"
        if not os.path.exists(weights_path):
            raise FileNotFoundError(f"File {weights_path} does not exist.")
        else:
            freeze_model(model, weights_path)
        unfreeze_last(model)
        model = model.to(device)
    
    # Applying Jacobian Regularization
    if args.jacobian:
        jacobian_reg = JacobianReg(args.n_proj)
    
    # Smart Regularizer instantiation
    if args.smart:
        smart_loss = SMARTLoss(model.forward_with_embeddings,kl_loss, loss_last_fn = sym_kl_loss, num_steps=args.num_steps, step_size=args.step_size_sm, epsilon=args.epsilon_sm, noise_var=args.noise_var_sm)

    # Applying LoRA
    if args.lora:
        freeze_all_but_last(model)
        model = replace_linear_with_lora(model, args.rank, args.alpha)
    
    if args.qlora:
        model = replace_linear_with_qlora(model, args.q_rank, args.q_alpha)
        unfreeze_last(model)

    model.to(device)
    

    lr = args.lr
    optimizer = AdamW(model.parameters(), lr=lr, weight_decay=args.weight_decay)
    scheduler = ReduceLROnPlateau(optimizer, patience=10)
    if args.verbose:
        print_requires_grad(model)
    # Run for the specified number of epochs.
    last_epoch = 0
    for epoch in range(args.epochs):
        model.train()
        train_loss = 0
        num_batches = 0
        perplexity = 0
        jacobian_train_loss = 0
        smart_train_loss = 0
        I = 0
        for batch in tqdm(sonnet_dataloader, desc=f'train-{epoch}', disable=TQDM_DISABLE):
            # Get the input and move it to the gpu (I do not recommend training this model on CPU).
            b_ids, b_mask = batch['token_ids'], batch['attention_mask']
            b_ids = b_ids.to(device)
            b_mask = b_mask.to(device)
            if args.debug:
                if I == 10:
                    break
                I += 1
            # Compute the loss, gradients, and update the model's parameters.
            optimizer.zero_grad()
            if args.jacobian or args.smart:
                x_embed = model.get_embeddings(b_ids)
                enforce_grad = not x_embed.requires_grad
                if enforce_grad:
                    x_embed.requires_grad_(True)
                logits = model.forward_with_embeddings(x_embed, b_mask)
            else:
                logits = model(b_ids, b_mask)

            logits = rearrange(logits[:, :-1].contiguous(), 'b t d -> (b t) d')  # Ignore the last prediction in the sequence.
            labels = b_ids[:, 1:].contiguous().flatten()  # Ignore the first token to compose the labels.
            loss = F.cross_entropy(logits, labels, reduction='mean')
            perplexity += torch.exp(loss).item()
            if args.jacobian:
                jacobian_lss = jacobian_reg(x_embed, logits)
                loss += args.jreg_lambda * jacobian_lss
                jacobian_train_loss += jacobian_lss.item()
            if args.smart:
                sm_loss = smart_loss(x_embed, logits,reshape_required=True, attn_masks=b_mask)
                smart_train_loss += sm_loss.item()
            
                loss += args.smart_lambda * sm_loss
            
            loss.backward()
            optimizer.step()

            train_loss += loss.item()
            num_batches += 1

        current_lr = optimizer.param_groups[0]['lr']
        train_loss = train_loss / num_batches
        perplexity = perplexity / num_batches
        jacobian_train_loss = jacobian_train_loss / num_batches
        smart_train_loss = smart_train_loss / num_batches
        writer.add_scalar('Loss/train', train_loss, epoch)
        writer.add_scalar('Perplexity/train', perplexity, epoch)
        if args.jacobian:
            writer.add_scalar('Jacobian/train', jacobian_train_loss, epoch)
        if args.smart:
            writer.add_scalar('SMART/train', smart_train_loss, epoch)
        writer.add_scalar('Learning Rate', current_lr, epoch)
        writer.flush()
 
        print(f"Epoch {epoch}: train loss :: {train_loss :.3f}.")
        print('Generating several output sonnets...')
        model.eval()
        for batch in held_out_sonnet_dataset:
            encoding = model.tokenizer(batch[1], return_tensors='pt', padding=True, truncation=True).to(device)
            output = model.generate(encoding['input_ids'], temperature=args.temperature, top_p=args.top_p, debug=args.debug)
            if args.verbose:
                print(f'{batch[1]}{output[1]}\n\n')

        # TODO: consider a stopping condition to prevent overfitting on the small dataset of sonnets.
        early_stopping(train_loss, model)
        scheduler.step(train_loss)
        if early_stopping.early_stop:
            print("Early stopping")
            break
        
        save_model_to = save_model_dir + f'/sonnet_{experiment_id}_{epoch}.pt'
        save_model(model, optimizer, args, save_model_to)
        last_epoch = epoch
        if epoch % 5 == 0:
            keep_latest_epoch_checkpoint(args.filepath.replace('.pt', '/'), epoch)
        
    metrics= {
        'experiment_id': experiment_id,
        'epochs': args.epochs,
        'lr': args.lr,
        'batch_size': args.batch_size,
        'model_size': args.model_size,
        'j_reg': args.jacobian,
        'jreg_lambda': args.jreg_lambda,
        'n_proj': args.n_proj,
        'lora': args.lora,
        'rank': args.rank,
        'alpha': args.alpha,    
        'qlora': args.qlora,
        'q_rank': args.q_rank,
        'q_alpha': args.q_alpha,
        'top_percent': args.top_percent,
        'spectrum': args.spectrum,
        'smart': args.smart,
        'smart_lambda': args.smart_lambda,
        'loss_train': train_loss,
        'perplexity_train': perplexity,
        'last_epoch': last_epoch, 
        'dropout': args.dropout,
        'attn_dropout': args.attn_dropout,
        'temperature': args.temperature,
        'weight_decay': args.weight_decay
        }
    return metrics



@torch.no_grad()
def generate_submission_sonnets(args, experiment_id, last_epoch=None, debug=False):
    device = torch.device('cuda') if args.use_gpu else torch.device('cpu')
    load_file_path = args.filepath.replace('.pt', f'/sonnet_{experiment_id}_{last_epoch}.pt')
    
    saved = torch.load(load_file_path, weights_only=False) #T

    model = SonnetGPT(saved['args'])
    if args.lora:
        freeze_all_but_last(model)
        model = replace_linear_with_lora(model, args.rank, args.alpha)

    if args.qlora:
        model = load_qlora_state_dict(model, saved['model'])
    else:
        model.load_state_dict(saved['model'])
    model = model.to(device)
    model.eval()

    # Create the held-out dataset: these only have the first 3 lines. Your job is to fill in the rest!
    held_out_sonnet_dataset = SonnetsDataset(args.held_out_sonnet_path)
    I=0
    generated_sonnets = []
    for batch in held_out_sonnet_dataset:
        if args.debug:
            if I == 10:
                break
            I +=1
        sonnet_id = batch[0]
        encoding = model.tokenizer(batch[1], return_tensors='pt', padding=False, truncation=True).to(device)
        output = model.generate(encoding['input_ids'], temperature=args.temperature, top_p=args.top_p)[0][0]
        decoded_output = model.tokenizer.decode(output)
        full_sonnet = f'{decoded_output}\n\n'
        generated_sonnets.append((sonnet_id, full_sonnet))
        if args.verbose:
            print(f'{decoded_output}\n\n')
    file_out = args.sonnet_out.replace('predictions/', "" )
    store_sonnets_path = args.filepath.replace('.pt', f'/{file_out}')
    with open(store_sonnets_path, "w+") as f:
        f.write(f"--Generated Sonnets-- \n\n")
        for sonnet in generated_sonnets:
            f.write(f"\n{sonnet[0]}\n")
            f.write(sonnet[1])


def get_args():
    parser = argparse.ArgumentParser()

    parser.add_argument("--sonnet_path", type=str, default="data/sonnets.txt")
    parser.add_argument("--held_out_sonnet_path", type=str, default="data/sonnets_held_out.txt")
    parser.add_argument("--sonnet_out", type=str, default="predictions/generated_sonnets.txt")

    parser.add_argument("--seed", type=int, default=11711)
    parser.add_argument("-e","--epochs", type=int, default=30)
    parser.add_argument("--use_gpu", action='store_true', default=True)
    parser.add_argument("--weight_decay", type=float, default=0.01)

    # Generation parameters.
    parser.add_argument("--temperature", type=float, help="softmax temperature.", default=1.2)
    parser.add_argument("--top_p", type=float, help="Cumulative probability distribution for nucleus sampling.",
                                            default=0.9)

    parser.add_argument("-b","--batch_size", help='The training batch size.', type=int, default=8)
    parser.add_argument("--lr", type=float, help="learning rate", default=1e-4) # LORA NEEDS A LEARNING RATE OF 1E
    parser.add_argument("--model_size", type=str, help="The model size as specified on hugging face.",
                                            choices=['gpt2', 'gpt2-medium', 'gpt2-large', 'gpt2-xl'], default='gpt2')
    
    # Jacobian Regularization Parameters
    parser.add_argument("--jacobian", action='store_true')
    parser.add_argument("--jreg_lambda", type=float, default=1e-4)
    parser.add_argument("--n_proj", type=int, default=1) # keep this as it is
    # LoRA parameters
    parser.add_argument("--lora", action='store_true')
    parser.add_argument("--rank", type=int, default=16)
    parser.add_argument("--alpha", type=int, default=16)
    ### QLoRA Parameters
    parser.add_argument("--qlora", action='store_true')
    parser.add_argument("--q_rank", type=int, default=8)
    parser.add_argument("--q_alpha", type=int, default=16)
    # Spectrum Parameters
    parser.add_argument("--spectrum", action='store_true')
    parser.add_argument("--top_percent", type=int, default=25)
    # SMART regularizer Parameters
    parser.add_argument("--smart", action='store_true')  
    parser.add_argument("--num_steps", type=int, default=1)
    parser.add_argument("--step_size_sm", type=float, default=1e-3)
    parser.add_argument("--epsilon_sm", type=float, default=1e-6)
    parser.add_argument("--noise_var_sm", type=float, default=1e-5)
    parser.add_argument("--smart_lambda", type=float, default=1e-4)
    ### Early Stopping Patience
    parser.add_argument("--patience", type=int, default=20)
    parser.add_argument("--delta", type=float, default=1e-4)
    ## Dropout Parameter Experiments
    parser.add_argument("--change_dropout", action='store_true')
    parser.add_argument("--dropout", type=float, default=0.1)
    parser.add_argument('--attn_dropout', type=float, default=0.)
    parser.add_argument('--debug', action='store_true')
    parser.add_argument('--verbose', action='store_true')

    args = parser.parse_args()
    return args


def add_arguments(args):
    """Add arguments that are deterministic on model size."""
    if args.model_size == 'gpt2':
        args.d = 768
        args.l = 12
        args.num_heads = 12
    elif args.model_size == 'gpt2-medium':
        args.d = 1024
        args.l = 24
        args.num_heads = 16
    elif args.model_size == 'gpt2-large':
        args.d = 1280
        args.l = 36
        args.num_heads = 20
    else:
        raise Exception(f'{args.model_size} is not supported.')
    return args


if __name__ == "__main__":
    experiment_id = generate_experiment_id()
    args = get_args()
    ### Fixing Paths for Dev ###
    args.held_out_sonnet_path = 'data/sonnets_held_out_dev.txt'
    args.sonnet_out = 'predictions/generated_sonnets_dev.txt'
    args.sonnet_out.replace('predictions/', "" )
    ## Generating Experiment ID and Modifying Paths
    os.makedirs('experiments/sonnet/', exist_ok=True)
    model_path = f'sonnet/{experiment_id}.pt'
    args.filepath =  os.path.join('experiments', model_path)
    seed_everything(args.seed) 
    ## Train Model
    metrics = train(args, experiment_id)
    # Generate Dev Sonnets
    generate_submission_sonnets(args, experiment_id, last_epoch=metrics['last_epoch'])
    store_txt_experiment_data(metrics, 'sonnet')
    keep_latest_epoch_checkpoint(args.filepath.replace('.pt', '/'), metrics['last_epoch'])
    ## Generate Test Sonnets and Store Metrics
    args.held_out_sonnet_path = "data/sonnets_held_out.txt"
    args.sonnet_out = "predictions/generated_sonnets.txt"
    generate_submission_sonnets(args, experiment_id, last_epoch=metrics['last_epoch'])
    print('Metrics have been stored in experiments/sonnet_metrics.txt')