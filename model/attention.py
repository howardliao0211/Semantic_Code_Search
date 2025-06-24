from torch import nn
import torch.nn.functional as F
import torch

class AdditiveAttention(nn.Module):

    def __init__(self, query_size: int, key_size: int, hidden_size: int, drop_p=0.1):
        super().__init__()
        self.query_linear = nn.Linear(query_size, hidden_size)
        self.key_linear = nn.Linear(key_size, hidden_size)
        self.value_linear = nn.Linear(hidden_size, 1)
        self.drop_out = nn.Dropout(drop_p)
    
    def forward(self, query: torch.Tensor, key: torch.Tensor, value: torch.Tensor):
        # q: batch, # of query, q_dim
        # k: batch, # of kv, k_dim
        # v: batch, # of kv, v_dim

        query_out: torch.Tensor = self.query_linear(query)    # batch, # of query, hidden
        key_out: torch.Tensor = self.key_linear(key)          # batch, # of kv, hidden

        # Use broadcast to add two matrix with different dimension.
        # batch, # of query, # of kv, hidden
        feature = F.tanh(query_out.unsqueeze(2) + key_out.unsqueeze(1))
        
        # Drop the last dimension, which is a 1.
        # batch, # of query, # of kv
        attention_score = self.value_linear(feature).squeeze(-1)
        attention_weight = F.softmax(attention_score, dim=-1)

        # Compute context vector
        # v: batch, # of kv, v_dim
        # attention_weight: batch, # of query, # of kv
        # context: batch, # of query, v_dim
        context = torch.bmm(self.drop_out(attention_weight), value)

        return context, attention_weight

if __name__ == '__main__':
    batch_size = 32
    num_queries = 10
    num_kv = 15
    query_size = 64
    key_size = 64
    value_size = 128
    hidden_size = 32

    # Initialize attention module
    attention = AdditiveAttention(query_size, key_size, hidden_size, drop_p=0.1)

    # Dummy input
    query = torch.randn(batch_size, num_queries, query_size)
    key = torch.randn(batch_size, num_kv, key_size)
    value = torch.randn(batch_size, num_kv, value_size)

    # Run forward pass
    context, attn_weights = attention(query, key, value)

    # Check output shapes
    assert context.shape == (batch_size, num_queries, value_size), f"context.shape: {context.shape}"
    assert attn_weights.shape == (batch_size, num_queries, num_kv), f"attn_weights.shape: {attn_weights.shape}"

    # Check attention weights sum to 1
    attn_sum = attn_weights.sum(dim=-1)
    assert torch.allclose(attn_sum, torch.ones_like(attn_sum), atol=1e-5), f"attn_weights do not sum to 1:\n{attn_sum}"

    print("✅ AdditiveAttention test passed!")
