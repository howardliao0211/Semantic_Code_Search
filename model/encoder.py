from torch import nn
import torch.nn.functional as F
import torch

class RNNEncoder(nn.Module):
    def __init__(self, input_size, hidden_size, dropout_p=0.1):
        super(RNNEncoder, self).__init__()
        self.hidden_size = hidden_size

        self.embedding = nn.Embedding(input_size, hidden_size)
        self.gru = nn.GRU(hidden_size, hidden_size, batch_first=True)
        self.dropout = nn.Dropout(dropout_p)

    def forward(self, input):
        embedded = self.dropout(self.embedding(input))
        output, hidden = self.gru(embedded)
        return output, hidden

if __name__ == '__main__':
    # construct input with size (batch, seq)
    batch_size, seq_size = 64, 16
    num_layer, input_size, hidden_size = 5, 1024, 8
    input = torch.randint(1, input_size, (batch_size, seq_size), dtype=torch.int)
    emb = nn.Embedding(input_size, hidden_size)
    gru = nn.GRU(hidden_size, hidden_size, num_layer, batch_first=True)

    # Input -> Embedding
    emb_out = emb(input)
    print('-----------------------')
    print(f'emb_out expected shape: {batch_size}, {seq_size}, {hidden_size}')
    print(f'emb_out shape: {emb_out.shape}')

    # Embedding -> GRU out, hidden
    gru_out, hidden = gru(emb_out)
    print('-----------------------')
    print(f'gru_out expected shape: {batch_size}, {seq_size}, {hidden_size}')
    print(f'gru_out shape: {gru_out.shape}')
    print('-----------------------')
    print(f'hidden expected shape: {num_layer}, {batch_size}, {hidden_size}')
    print(f'hidden shape: {hidden.shape}')

