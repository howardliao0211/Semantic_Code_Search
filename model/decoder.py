from torch import nn
from typing import Any, Protocol
from .attention import AdditiveAttention
import torch.nn.functional as F
import torch

class SequenceDecoder(Protocol):

    def forward(self, encoder_output: torch.Tensor, encoder_hidden: torch.Tensor, tgt_tensor: torch.Tensor|Any=None) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        pass


class RNNDecoder(nn.Module):

    def __init__(self, hidden_size: int, output_size: int, bos_token: int, eos_token: int) -> None:
        super(RNNDecoder, self).__init__()

        self.emb = nn.Embedding(output_size, hidden_size)
        self.gru = nn.GRU(hidden_size, hidden_size, batch_first=True)
        self.out = nn.Linear(hidden_size, output_size)
        self.bos_token = bos_token
        self.eos_token = eos_token

    def forward(self, encoder_output: torch.Tensor, encoder_hidden: torch.Tensor, tgt_tensor: torch.Tensor|Any=None) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        batch_size, seq_size = encoder_output.size(0), encoder_output.size(1)
        decoder_input = torch.empty(batch_size, 1, dtype=torch.long, device=encoder_output.device).fill_(self.bos_token)

        decoder_outputs = []
        decoder_hidden = encoder_hidden

        max_len = tgt_tensor.size(1) if tgt_tensor is not None else seq_size
        
        for seq_idx in range(max_len):
            decoder_output, decoder_hidden = self._forward_step(decoder_input, decoder_hidden)
            decoder_outputs.append(decoder_output)

            if tgt_tensor is not None:
                # replace the next decoder input with the tgt_tensor (shape: batch, seq_size)
                # use unsqueeze because tgt_tensor[:, seq_idx] will return a tensor with shape: (batch_size,)
                decoder_input = tgt_tensor[:, seq_idx].unsqueeze(1)
            else:
                # use the decoder's own output as the next input.
                # decoder's output shape: (batch, 1, output_size)
                # we need to squeeze the last dimension so that the decoder_input shape will still be (batch, 1)
                out_value, out_idx = decoder_output.topk(k=1, dim=-1)
                
                # need to detach so that gradient will not explode or vanish.
                decoder_input = out_idx.squeeze(-1).detach()

                # If the model predict eos_token, then stop predicting.
                if (decoder_input == self.eos_token).all():
                    break
        
        # decoder_outputs will be a list of tensor with shape (batch, 1, output_size)
        decoder_outputs = torch.cat(decoder_outputs, dim=1)
        decoder_outputs = F.log_softmax(decoder_outputs, dim=-1)
        
        # Append a None so that the output of the decoder would be the same as the attention decoder. 
        return decoder_outputs, decoder_hidden, None

    def _forward_step(self, input, hidden) -> tuple[torch.Tensor, torch.Tensor]:
        emb_out = F.relu(self.emb(input))
        gru_out, hidden_state = self.gru(emb_out, hidden)
        lin_out = self.out(gru_out)
        return lin_out, hidden_state

class BahdanauAttentionDecoder(nn.Module):
    
    def __init__(self, hidden_size, output_size, bos_token, eos_token, drop_p = 0.1):
        super(BahdanauAttentionDecoder, self).__init__()

        self.attention = AdditiveAttention(query_size=hidden_size,
                                           key_size=hidden_size,
                                           hidden_size=hidden_size,
                                           drop_p=drop_p)

        self.emb = nn.Embedding(output_size, hidden_size)
        self.rnn = nn.GRU(2 * hidden_size, hidden_size, batch_first=True)
        self.out = nn.Linear(hidden_size, output_size)
        self.dropout = nn.Dropout(drop_p)
        self.bos_token = bos_token
        self.eos_token = eos_token
    
    def forward(self, encoder_outputs: torch.Tensor, encoder_hidden: torch.Tensor, tgt_tensor: torch.Tensor|Any=None):
        batch_size, seq_size = encoder_outputs.size(0), encoder_outputs.size(1)
        decoder_input = torch.empty(batch_size, 1, dtype=torch.long, device=encoder_outputs.device).fill_(self.bos_token)

        decoder_outputs = []
        attention_weights = []
        decoder_hidden = encoder_hidden

        max_len = tgt_tensor.size(1) if tgt_tensor is not None else seq_size

        for seq_idx in range(max_len):
            decoder_output, decoder_hidden, attention_weight = self._forward_step(
                decoder_input, decoder_hidden, encoder_outputs
            )
            decoder_outputs.append(decoder_output)
            attention_weights.append(attention_weight)

            if tgt_tensor is not None:
                # replace the next decoder input with the tgt_tensor (shape: batch, seq_size)
                # use unsqueeze because tgt_tensor[:, seq_idx] will return a tensor with shape: (batch_size,)
                decoder_input = tgt_tensor[:, seq_idx].unsqueeze(1)
            else:
                # use the decoder's own output as the next input.
                # decoder's output shape: (batch, 1, output_size)
                out_value, out_idx = decoder_output.topk(k=1, dim=-1)
                
                # we need to squeeze the last dimension so that the decoder_input shape will still be (batch, 1)
                # need to detach so that gradient will not explode or vanish.
                decoder_input = out_idx.squeeze(-1).detach()

                # If the model predict eos_token, then stop predicting.
                if (decoder_input == self.eos_token).all():
                    break
        
        # decoder_outputs will be a list of tensor with shape (batch, 1, output_size)
        decoder_outputs = torch.cat(decoder_outputs, dim=1)
        attention_weights = torch.cat(attention_weights, dim=1) # (batch, # of layer, # of seq)
        decoder_outputs = F.log_softmax(decoder_outputs, dim=-1)
        
        # Append a None so that the output of the decoder would be the same as the attention decoder. 
        return decoder_outputs, decoder_hidden, attention_weights

    def _forward_step(self, decoder_input: torch.Tensor, hidden: torch.Tensor, encoder_outputs: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:

        # Compute embedding and drop out.
        # embedded: (# of batch, 1, hidden_size)
        embedded = self.dropout(self.emb(decoder_input))

        # Compute context vector.
        # query: decoder hidden state (# of layer, # of batch, hidden_size)
        # key:   encoder output (# of batch, # of seq, hidden_size)
        # value: encoder output (# of batch, # of seq, hidden_size)

        # we need to swap decoder hidden state's first and second axis.
        query = hidden.permute(1, 0, 2)
        key = encoder_outputs
        value = encoder_outputs

        # context shape: (# of batch, # of layer, emb_size)
        # attention_w shape: (# of batch, # of layer, # of seq)
        context, attention_weight = self.attention(query, key, value)

        # concat the embedding with context
        embedded = torch.cat((embedded, context), dim=-1)
        gru_out, hidden_state = self.rnn(embedded, hidden)
        lin_out = self.out(gru_out)

        return lin_out, hidden_state, attention_weight

if __name__ == '__main__':
    hidden_size = 64
    output_size = 128
    decoder = RNNDecoder(hidden_size, output_size)

    encoder_batch_size = 64
    encoder_seq_size = 16
    encoder_hidden_layer_size = 1

    encoder_output = torch.rand(encoder_batch_size, encoder_seq_size, hidden_size)
    encoder_hidden = torch.rand(encoder_hidden_layer_size, encoder_batch_size, hidden_size)

    decoder_output, decoder_hidden = decoder.forward(encoder_output, encoder_hidden, bos_token=0)
    assert decoder_output.shape == torch.Size((encoder_batch_size, encoder_seq_size, output_size)), f'decoder_output.shape: {decoder_output.shape}'
    assert decoder_hidden.shape == encoder_hidden.shape

