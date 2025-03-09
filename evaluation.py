# !/usr/bin/env python3

"""
Evaluation code for Quora paraphrase detection.

model_eval_paraphrase is suitable for the dev (and train) dataloaders where the label information is available.
model_test_paraphrase is suitable for the test dataloader where label information is not available.
"""

import torch
import torch.nn.functional as F
from sklearn.metrics import f1_score, accuracy_score
from tqdm import tqdm
import numpy as np
from sacrebleu.metrics import CHRF
from datasets import (
  SonnetsDataset,
)

TQDM_DISABLE = False
DEBUGGING = False

@torch.no_grad()
def model_eval_paraphrase(dataloader, model, device, return_loss=False):
  model.eval()  # Switch to eval model, will turn off randomness like dropout.
  y_true, y_pred, sent_ids = [], [], []
  I = 0
  loss = 0
  for step, batch in enumerate(tqdm(dataloader, desc=f'eval', disable=TQDM_DISABLE)):
    b_ids, b_mask, b_sent_ids, labels = batch['token_ids'], batch['attention_mask'], batch['sent_ids'], batch[
      'labels'].flatten()
    if DEBUGGING:
      if I == 10: # FOR TESTING
        break
      I +=1
    b_ids = b_ids.to(device)
    b_mask = b_mask.to(device)
    logits = model(b_ids, b_mask)
    
    logits = model(b_ids, b_mask)
    labels_i = labels.to(device)
    labels_i = torch.where(labels_i == 8505, torch.tensor(1, device=device), torch.tensor(0, device=device))
    loss += F.cross_entropy(logits, labels_i, reduction='mean')
    logits = logits.cpu().numpy()
    preds = np.argmax(logits, axis=1).flatten()

    y_true.extend(labels_i.cpu().numpy())
    y_pred.extend(preds)
    sent_ids.extend(b_sent_ids)
  loss = loss / len(dataloader)
  f1 = f1_score(y_true, y_pred, average='macro')
  acc = accuracy_score(y_true, y_pred)
  if return_loss:
    return acc, f1, y_pred, y_true, sent_ids, loss
  else:
    return acc, f1, y_pred, y_true, sent_ids


@torch.no_grad()
def model_test_paraphrase(dataloader, model, device):
  model.eval()  # Switch to eval model, will turn off randomness like dropout.
  y_true, y_pred, sent_ids = [], [], []
  for step, batch in enumerate(tqdm(dataloader, desc=f'eval', disable=TQDM_DISABLE)):
    b_ids, b_mask, b_sent_ids = batch['token_ids'], batch['attention_mask'], batch['sent_ids']

    b_ids = b_ids.to(device)
    b_mask = b_mask.to(device)

    logits = model(b_ids, b_mask).cpu().numpy()
    preds = np.argmax(logits, axis=1).flatten()

    y_pred.extend(preds)
    sent_ids.extend(b_sent_ids)

  return y_pred, sent_ids


def test_sonnet(
    test_path='predictions/generated_sonnets.txt',
    gold_path='data/TRUE_sonnets_held_out.txt'
):
    chrf = CHRF()

    # get the sonnets
    generated_sonnets = [x[1] for x in SonnetsDataset(test_path)]
    true_sonnets = [x[1] for x in SonnetsDataset(gold_path)]
    max_len = min(len(true_sonnets), len(generated_sonnets))
    true_sonnets = true_sonnets[:max_len]
    generated_sonnets = generated_sonnets[:max_len]

    # compute chrf
    chrf_score = chrf.corpus_score(generated_sonnets, [true_sonnets])
    return float(chrf_score.score)