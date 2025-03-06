'''
Paraphrase detection for GPT starter code.

Consider:
 - ParaphraseGPT: Your implementation of the GPT-2 classification model.
 - train: Training procedure for ParaphraseGPT on the Quora paraphrase detection dataset.
 - test: Test procedure. This function generates the required files for your submission.

Running:
  `python paraphrase_detection.py --use_gpu`
trains and evaluates your ParaphraseGPT model and writes the required submission files.
'''

from datetime import datetime
import os
import random
import argparse
import random
import json

import numpy as np
import torch.nn.functional as F
import torch
from torch import nn
from torch.utils.data import DataLoader
from tqdm import tqdm
from torch.utils.tensorboard import SummaryWriter

from datasets import (
  ParaphraseDetectionDataset,
  ParaphraseDetectionTestDataset,
  load_paraphrase_data
)
# Local Imports
from evaluation import model_eval_paraphrase, model_test_paraphrase
from models.gpt2 import GPT2Model
from extensions.lora_layer import replace_linear_with_lora, freeze_all_but_last
from extensions.spectrum import freeze_model, unfreeze_last
from extensions.jacobian_reg import JacobianReg
from extensions.pipeline_utils import store_txt_experiment_data,    generate_experiment_id
from extensions.smart_pytorch import SMARTLoss
from extensions.smart_loss import kl_loss, sym_kl_loss
from extensions.dropout_modifier import modify_model_dropout
from extensions.early_stopper import EarlyStopping
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


class ParaphraseGPT(nn.Module):
  """Your GPT-2 Model designed for paraphrase detection."""

  def __init__(self, args):
    super().__init__()
    self.gpt = GPT2Model.from_pretrained(model=args.model_size, d=args.d, l=args.l, num_heads=args.num_heads)
    self.paraphrase_detection_head = nn.Linear(args.d, 2)  # Paraphrase detection has two outputs: 1 (yes) or 0 (no).

    # By default, fine-tune the full model.
    for param in self.gpt.parameters():
      param.requires_grad = True

  def forward(self, input_ids, attention_mask, return_embeddings=False):
    """
    TODO: Predict the label of the token using the paraphrase_detection_head Linear layer.

    We structure the input as:

      'Is "{s1}" a paraphrase of "{s2}"? Answer "yes" or "no": '

    So you want to find the prediction for the next token at the end of this sentence. Optimistically, it will be the
    token "yes" (byte pair encoding index of 8505) for examples that are paraphrases or "no" (byte pair encoding index
     of 3919) for examples that are not paraphrases.
    """

    'Takes a batch of sentences and produces embeddings for them.'
    outputs = self.gpt(input_ids=input_ids, attention_mask=attention_mask, return_embeddings=return_embeddings)
    last_hidden_state = outputs['last_hidden_state']
    logits = self.paraphrase_detection_head(last_hidden_state[:, -1, :])  
    if return_embeddings:
      return logits, outputs['embeddings']
    else:
      return logits


def save_model(model, optimizer, args, filepath):
  save_info = {
    'model': model.state_dict(),
    'optim': optimizer.state_dict(),
    'args': args,
    'system_rng': random.getstate(),
    'numpy_rng': np.random.get_state(),
    'torch_rng': torch.random.get_rng_state(),
  }

  torch.save(save_info, filepath)
  print(f"save the model to {filepath}")


def train(args, experiment_id=1):
  """Train GPT-2 for paraphrase detection on the Quora dataset."""
  device = torch.device('cuda') if args.use_gpu and torch.cuda.is_available() else torch.device('cpu')
  experiment_path = args.file_path.replace('.pt', '').replace('experiments/', 'runs/')
  writer = SummaryWriter(f'{experiment_path}_{experiment_id}')
  # Create the data and its corresponding datasets and dataloader.
  para_train_data = load_paraphrase_data(args.para_train)
  para_dev_data = load_paraphrase_data(args.para_dev)

  para_train_data = ParaphraseDetectionDataset(para_train_data, args)
  para_dev_data = ParaphraseDetectionDataset(para_dev_data, args)

  para_train_dataloader = DataLoader(para_train_data, shuffle=True, batch_size=args.batch_size,
                                     collate_fn=para_train_data.collate_fn)
  para_dev_dataloader = DataLoader(para_dev_data, shuffle=False, batch_size=args.batch_size,
                                   collate_fn=para_dev_data.collate_fn)

  args = add_arguments(args)
  model = ParaphraseGPT(args)
 # Adding change in dropout rates
  if args.change_dropout:
      modify_model_dropout(model, args.dropout, args.attn_dropout)
  
  # Early stopping
  early_stopping = EarlyStopping(patience=args.patience, delta=args.delta)
  
  # Applying LoRA
  if args.lora:
      freeze_all_but_last(model)
      model = replace_linear_with_lora(model, args.rank, args.alpha)
      print(model)

  model = model.to(device)
  # Applying Spectrum
  if args.spectrum:
      weights_path = f"extensions/spectrum/model_snr_results/snr_results_gpt2_unfrozenparameters_{args.top_percent}percent.yaml"
      if not os.path.exists(weights_path):
          raise FileNotFoundError(f"File {weights_path} does not exist.")
      else:
          freeze_model(model, weights_path)
      unfreeze_last(model)
      print(model)
      model = model.to(device)
  
  # Applying Jacobian Regularization
  if args.jacobian:
      jacobian_reg = JacobianReg(args.n_proj)
  
  # Smart Regularizer instantiation
  if args.smart:
      smart_loss = SMARTLoss(model.forward_with_embeddings,kl_loss, loss_last_fn = sym_kl_loss, num_steps=args.num_steps, step_size=args.step_size_sm, epsilon=args.epsilon_sm, noise_var=args.noise_var_sm)

  lr = args.lr
  optimizer = AdamW(model.parameters(), lr=lr, weight_decay=0.)
  best_dev_acc = 0


  # Run for the specified number of epochs.
  for epoch in range(args.epochs):
    model.train()
    train_loss = 0
    num_batches = 0
    accuracy = 0
    perplexity = 0
    for batch in tqdm(para_train_dataloader, desc=f'train-{epoch}', disable=TQDM_DISABLE):
      # Get the input and move it to the gpu (I do not recommend training this model on CPU).
      b_ids, b_mask, labels = batch['token_ids'], batch['attention_mask'], batch['labels'].flatten()
      b_ids = b_ids.to(device)
      b_mask = b_mask.to(device)
      labels = labels.to(device)

      # Compute the loss, gradients, and update the model's parameters.
      optimizer.zero_grad()
      if args.jacobian or args.smart:
         logits, x_embed = model(b_ids, b_mask, return_embeddings=True)
      else:
        logits = model(b_ids, b_mask)
      preds = torch.argmax(logits, dim=1)
      labels = torch.where(labels == 8505, torch.tensor(1, device=device), torch.tensor(0, device=device))
      loss = F.cross_entropy(logits, labels, reduction='mean')
      perplexity += torch.exp(loss).item()
      
      if args.jacobian:
          jacobian_lss = jacobian_reg(x_embed, logits)
          loss += args.jreg_lambda * jacobian_lss
      if args.smart:
          sm_loss = smart_loss(x_embed, logits,reshape_required=True, attn_masks=b_mask)
          loss += args.smart_lambda * sm_loss
      accuracy += (preds == labels).sum().item()
      

      loss.backward()
      optimizer.step()

      train_loss += loss.item()
      num_batches += 1

    train_loss = train_loss / num_batches
    accuracy = accuracy / len(para_train_data)
    dev_acc, dev_f1, *_ = model_eval_paraphrase(para_dev_dataloader, model, device)
    ## Tensoboard computations
    perplexity = perplexity / num_batches
    writer.add_scalar("Train/Loss", train_loss, epoch)
    writer.add_scalar("Train/Accuracy", accuracy, epoch)
    writer.add_scalar("Train/Perplexity", perplexity, epoch)
    writer.add_scalar("Dev/Accuracy", dev_acc, epoch)
    writer.add_scalar("Dev/F1", dev_f1, epoch)


    if dev_acc > best_dev_acc:
      best_dev_acc = dev_acc
      save_model(model, optimizer, args, args.filepath) #TODO: MODIFY THIS SAVE FILE PATH

    ## Applying early stopping
    if early_stopping.step(dev_acc):
        print(f"Early Stopping at epoch {epoch}")
        break

    print(f"Epoch {epoch}: train loss :: {train_loss :.3f}, dev acc :: {dev_acc :.3f}")

  metrics= {
    'experiment_id': experiment_id,
    "best_dev_acc": best_dev_acc,
    "best_dev_f1": dev_f1,
    'epochs': args.epochs,
    'lr': args.lr,
    'batch_size': args.batch_size,
    'model_size': args.model_size,
    'j_reg': args.j_reg,
    'n_proj': args.n_proj,
    'rank': args.rank,
    'alpha': args.alpha,
    'lora': args.lora,
    'top_percent': args.top_percent,
    'spectrum': args.spectrum,
    'smart': args.smart,
    'smart_lambda': args.smart_lambda,
    'loss': train_loss,
    'accuracy': accuracy,
    'perplexity': perplexity
    }
  return metrics

@torch.no_grad()
def test(args, metrics=None):
  """Evaluate your model on the dev and test datasets; save the predictions to disk."""
  # device = torch.device('cuda') if args.use_gpu else torch.device('cpu')
  if args.use_gpu:
    if torch.cuda.is_available():
        device = torch.device("cuda")
    elif torch.backends.mps.is_available():
        device = torch.device("mps")
    else:
        device = torch.device("cpu")
  else:
      device = torch.device("cpu")

  saved = torch.load(args.filepath)

  model = ParaphraseGPT(saved['args'])
  model.load_state_dict(saved['model'])
  model = model.to(device)
  model.eval()
  print(f"Loaded model to test from {args.filepath}")

  para_dev_data = load_paraphrase_data(args.para_dev)
  para_test_data = load_paraphrase_data(args.para_test, split='test')

  para_dev_data = ParaphraseDetectionDataset(para_dev_data, args)
  para_test_data = ParaphraseDetectionTestDataset(para_test_data, args)

  para_dev_dataloader = DataLoader(para_dev_data, shuffle=False, batch_size=args.batch_size,
                                   collate_fn=para_dev_data.collate_fn)
  para_test_dataloader = DataLoader(para_test_data, shuffle=True, batch_size=args.batch_size,
                                    collate_fn=para_test_data.collate_fn)

  dev_para_acc, _, dev_para_y_pred, _, dev_para_sent_ids = model_eval_paraphrase(para_dev_dataloader, model, device)
  print(f"dev paraphrase acc :: {dev_para_acc :.3f}")
  test_para_y_pred, test_para_sent_ids = model_test_paraphrase(para_test_dataloader, model, device)

  with open(args.para_dev_out, "w+") as f:
    f.write(f"id \t Predicted_Is_Paraphrase \n")
    for p, s in zip(dev_para_sent_ids, dev_para_y_pred):
      f.write(f"{p}, {s} \n")

  with open(args.para_test_out, "w+") as f:
    f.write(f"id \t Predicted_Is_Paraphrase \n")
    for p, s in zip(test_para_sent_ids, test_para_y_pred):
      f.write(f"{p}, {s} \n")
  if metrics is not None:
      metrics['dev_para_acc'] = dev_para_acc # TODO: ADD METRICS FOR THE  TEST SET
  return metrics



def get_args():
  parser = argparse.ArgumentParser()

  parser.add_argument("--para_train", type=str, default="data/quora-train.csv")
  parser.add_argument("--para_dev", type=str, default="data/quora-dev.csv")
  parser.add_argument("--para_test", type=str, default="data/quora-test-student.csv")
  parser.add_argument("--para_dev_out", type=str, default="predictions/para-dev-output.csv")
  parser.add_argument("--para_test_out", type=str, default="predictions/para-test-output.csv")
  parser.add_argument("--seed", type=int, default=11711)
  parser.add_argument("--epochs", type=int, default=10)
  parser.add_argument("--use_gpu", action='store_true')

  parser.add_argument("--batch_size", help='sst: 64, cfimdb: 8 can fit a 12GB GPU', type=int, default=8)
  parser.add_argument("--lr", type=float, help="learning rate", default=1e-5)
  parser.add_argument("--model_size", type=str,
                      help="The model size as specified on hugging face. DO NOT use the xl model.",
                      choices=['gpt2', 'gpt2-medium', 'gpt2-large'], default='gpt2')
  
  # Jacobian Regularization Parameters
  parser.add_argument("--jacobian", action='store_true')
  parser.add_argument("--jreg_lambda", type=float, default=0.1)
  parser.add_argument("--n_proj", type=int, default=1) # keep this as it is
  # LoRA parameters
  parser.add_argument("--lora", action='store_true')
  parser.add_argument("--rank", type=int, default=16)
  parser.add_argument("--alpha", type=int, default=16)
  # Spectrum Parameters
  parser.add_argument("--spectrum", action='store_true')
  parser.add_argument("--top_percent", type=int, default=25)
  # SMART regularizer Parameters
  parser.add_argument("--smart", action='store_true')  
  parser.add_argument("--num_steps", type=int, default=1)
  parser.add_argument("--step_size_sm", type=float, default=1e-3)
  parser.add_argument("--epsilon_sm", type=float, default=1e-6)
  parser.add_argument("--noise_var_sm", type=float, default=1e-5)
  parser.add_argument("--smart_lambda", type=float, default=1e-5)
  ### Early Stopping Patience
  parser.add_argument("--patience", type=int, default=5)
  parser.add_argument("--delta", type=float, default=1e-4)
  ## Dropout Parameter Experiments
  parser.add_argument("--change_dropout", action='store_true')
  parser.add_argument("--dropout", type=float, default=0.1)
  parser.add_argument('--attn_dropout', type=float, default=0.1)

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
  os.makedirs('experiments/paraphrase', exist_ok=True)
  model_path = f'paraphrase-{experiment_id}.pt'
  args.filepath =  os.path.join('experiments', model_path)
  seed_everything(args.seed)  # Fix the seed for reproducibility.
  metrics = train(args, experiment_id)
  metrics = test(args, metrics)
  store_txt_experiment_data(metrics, 'paraphrase')
  print('Metrics have been stored in experiments/paraphrase_metrics.txt')



       
      