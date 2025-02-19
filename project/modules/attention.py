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
    self.max_len = None
    self.attn_mask = None

  def transform(self, x, linear_layer):
    # The corresponding linear_layer of k, v, q are used to project the hidden_state (x).
    proj = linear_layer(x)
    # Next, we need to produce multiple heads for the proj. This is done by spliting the
    # hidden state to self.num_attention_heads, each of size self.attention_head_size.
    proj = rearrange(proj, 'b t (h d) -> b t h d', h=self.num_attention_heads)
    # By proper transpose, we have proj of size [bs, num_attention_heads, seq_len, attention_head_size].
    proj = rearrange(proj, 'b t h d -> b h t d')
    return proj

  def attention(self, key, query, value, attention_mask):
    "Method to calculate the multi-head attention."
    # query shape: [batch_size, num_heads, seq_len, attention_head_size]
    # key shape: [batch_size, num_heads, seq_len, attention_head_size]
    if self.max_len is None:
      self.max_len = key.shape[-2]
      self.attn_mask = torch.triu(torch.ones(self.max_len, self.max_len), 1).to(key.device)

    qk = torch.einsum('b h i d, b h j d -> b h i j', query, key)
    qk = torch.matmul(query, key.transpose(-1, -2))
    print(qk.shape)
  
    # Alternitavely 

    # attention mask should have a shape like attention_mask[:, None, None, :] (bs, 1, 1, seq_len)
    d_k = key.shape[-1]
    attn_w = qk.masked_fill(self.attn_mask[:key.size(-2), :key.size(-2)] == 1., float('-inf'))
    attn_w = F.softmax(attn_w/ (d_k ** 0.5) , dim=-1) # [batch_size, num_heads, seq_len, seq_len]
    attn_w = self.dropout(attn_w)

    # value shape: [batch_size, num_heads, seq_len, attention_head_size]
    # out = torch.matmul(attn_w, value)

    # Multiplying Attention weights with value
    out= torch.einsum('b h i j, b h j d -> b h i d', attn_w, value)
    
    # Concatenating Step
    out = rearrange(out, 'b h t d -> b t (h d)')
    # out = torch.cat(torch.split(out, 1, dim=1), dim=-1).squeeze(1).contiguous()
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
  
