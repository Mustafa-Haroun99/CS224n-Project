from torchtext.data.metrics import bleu_score

def calculate_bleu(references, hypotheses, max_n=4):
    """
    Calculate BLEU score for generated sonnets
    """
    weights = [1/max_n] * max_n  # Equal weights for 1-gram to max_n-gram
    return bleu_score(hypotheses, references, weights)