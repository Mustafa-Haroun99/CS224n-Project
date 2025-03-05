import torch

def modify_model_dropout(model, dropout_rate, attention_dropout_rate=None):
    """Specific implementation for transformer models"""
    for module in model.modules():
        if isinstance(module, torch.nn.Dropout):
            module.p = dropout_rate
        # Handle model-specific dropout attributes
        elif hasattr(module, 'dropout'):
            module.dropout.p = dropout_rate
        elif hasattr(module, 'attention_dropout'):
            module.attention_dropout.p = attention_dropout_rate
    