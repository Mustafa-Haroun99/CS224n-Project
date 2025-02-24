import torch
import torch.nn.functional as F
from einops import rearrange
from torch import nn


class CausalSelfAttention(nn.Module):
  def __init__(self, config):
    super().__init__()

    self.num_attention_heads = config.num_attention_heads
    self.attention_head_size = int(config.hidden_size / config.num_attention_heads)
    self.all_head_size = self.num_attention_heads * self.attention_head_size

    # Initialize the linear transformation layers for key, value, query.
    self.query = nn.Linear(config.hidden_size, self.all_head_size)
    self.key = nn.Linear(config.hidden_size, self.all_head_size)
    self.value = nn.Linear(config.hidden_size, self.all_head_size)
    # This dropout is applied to normalized attention scores following the original
    # implementation of transformer. Although it is a bit unusual, we empirically
    # observe that it yields better performance.
    self.dropout = nn.Dropout(config.attention_probs_dropout_prob)
    max_len = None
    attn_mask = None
    self.register_buffer("attn_mask", attn_mask, persistent=False)
    self.register_buffer("max_len", max_len, persistent=False)


  def transform(self, x, linear_layer):
    # The corresponding linear_layer of k, v, q are used to project the hidden_state (x).
    proj = linear_layer(x)
    # Next, we need to produce multiple heads for the proj. This is done by spliting the
    # hidden state to self.num_attention_heads, each of size self.attention_head_size.
    proj = rearrange(proj, 'b t (h d) -> b t h d', h=self.num_attention_heads)
    # By proper transpose, we have proj of size [bs, num_attention_heads, seq_len, attention_head_size].
    proj = rearrange(proj, 'b t h d -> b h t d')
    return proj

  def _register_masks(self, seq_len, device):
      max_len = seq_len
      self.max_len = torch.tensor(max_len, dtype=torch.int, device=device)
      self.attn_mask = torch.triu(torch.ones(max_len, max_len), 1,).to(device)
      
  def attention(self, key, query, value, attention_mask):
    "Method to calculate the multi-head attention."
    # query shape: [batch_size, num_heads, seq_len, attention_head_size]
    # key shape: [batch_size, num_heads, seq_len, attention_head_size]
    seq_len = key.size(-2)
    if self.attn_mask is None or self.max_len < seq_len:
        self._register_masks(seq_len, key.device)

    qk = torch.einsum('b h i d, b h j d -> b h i j', query, key)

    # attention mask should have a shape like attention_mask[:, None, None, :] (bs, 1, 1, seq_len)
    d_k = key.shape[-1]
    if attention_mask is None:
        attention_mask = 0
 
    attn_w = qk.masked_fill(self.attn_mask[:seq_len, :seq_len] == 1., float('-inf')) + attention_mask
    attn_w = self.dropout(F.softmax(attn_w/ (d_k ** 0.5) , dim=-1)) # [batch_size, num_heads, seq_len, seq_len]

    # value shape: [batch_size, num_heads, seq_len, attention_head_size]
    # Multiplying Attention weights with value
    out= torch.einsum('b h i j, b h j d -> b h i d', attn_w, value)
    
    # Concatenating Step
    out = rearrange(out, 'b h t d -> b t (h d)') 
  
    return out
  

  def forward(self, hidden_states, attention_mask):
    """
    hidden_states: [bs, seq_len, hidden_state]
    attention_mask: [bs, 1, 1, seq_len]
    output: [bs, seq_len, hidden_state]
    """
    # First, we have to generate the key, value, query for each token for multi-head attention
    # using self.transform (more details inside the function).
    # Size of *_layer is [bs, num_attention_heads, seq_len, attention_head_size].
    key_layer = self.transform(hidden_states, self.key)
    value_layer = self.transform(hidden_states, self.value)
    query_layer = self.transform(hidden_states, self.query)
    
    # Calculate the multi-head attention.
    attn_value = self.attention(key_layer, query_layer, value_layer, attention_mask)
    return attn_value
  
