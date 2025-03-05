def modify_transformer_dropout(model, new_dropout_rate):
    """Specific implementation for transformer models"""
    for module in model.modules():
        if isinstance(module, torch.nn.Dropout):
            module.p = new_dropout_rate
        # Handle model-specific dropout attributes
        elif hasattr(module, 'dropout'):
            module.dropout.p = new_dropout_rate
        elif hasattr(module, 'attention_dropout'):
            module.attention_dropout.p = new_dropout_rate
    
    return model