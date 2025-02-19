from torch import nn
import torch.nn.functional as F
import torch

from modules.attention import CausalSelfAttention

class GPT2Layer(nn.Module):
    def __init__(self, config):
        super().__init__()
        # Multi-head attention
        self.self_attention = CausalSelfAttention(config)
        # Add-norm for multi-head attention
        self.attention_dense = nn.Linear(config.hidden_size, config.hidden_size)
        self.attention_layer_norm = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)
        self.attention_dropout = nn.Dropout(config.hidden_dropout_prob)
        # Feed forward
        self.interm_dense = nn.Linear(config.hidden_size, config.intermediate_size)
        self.interm_af = F.gelu
        # Add-norm for feed forward
        self.out_dense = nn.Linear(config.intermediate_size, config.hidden_size)
        self.out_layer_norm = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)
        self.out_dropout = nn.Dropout(config.hidden_dropout_prob)

    def add(self, input, output, dense_layer, dropout):
        """
        Applies residual connection with dropout after a dense layer.
        """
        return input + dropout(dense_layer(output))

    def forward(self, hidden_states, attention_mask):
      """
      TODO: Implement the forward pass. Some key points to consider:
            - A multi-head attention layer (CausalSelfAttention) that computes self-attention based on masked inputs.
            - Layer normalization applied *before* the attention layer and feed-forward layer.
            - Apply dropout, residual connection, and layer normalization according to the plot in the assignment. (Use self.add)
            - A feed-forward layer that applies transformations to further refine the hidden states.
      """
      h_norm = self.attention_layer_norm(hidden_states)
      attn_output = self.self_attention(h_norm, attention_mask)
      attn_output = self.add(hidden_states, attn_output, self.attention_dense, self.attention_dropout)
      attn_output_n = self.attention_layer_norm(attn_output)
      inter_output = self.interm_af(self.interm_dense(attn_output_n))
      out_output = self.add(attn_output, inter_output, self.out_dense, self.out_dropout)
      output = self.out_layer_norm(out_output)
      return output
      ### MLP LAYERS



    ### YOUR CODE HERE
    