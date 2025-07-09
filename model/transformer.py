from torch import nn
from torch import nn
from typing import Optional
import torch
import math

class PositionEncoding(nn.Module):

    def __init__(self, seq_size, emb_size, dropout_p):
        super().__init__()

        self.position_emb_matrix = torch.zeros((1, seq_size, emb_size))
        position_matrix = torch.arange(seq_size, dtype=torch.float32).reshape(-1, 1) / \
            torch.pow(10000, torch.arange(0, emb_size, 2, dtype=torch.float32) / emb_size)

        self.position_emb_matrix[:, :, 0::2] = torch.sin(position_matrix)
        self.position_emb_matrix[:, :, 1::2] = torch.cos(position_matrix)
        self.dropout = nn.Dropout(dropout_p)

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        seq_size = input.size(1)
        input += self.position_emb_matrix[:, :seq_size, :].to(input.device)
        return self.dropout(input)

class AddNorm(nn.Module):

    def __init__(self, norm_shape: int, dropout_p: float):
        super().__init__()
        self.norm = nn.LayerNorm(norm_shape)
        self.dropout = nn.Dropout(dropout_p)
    
    def forward(self, input: torch.Tensor, weight: torch.Tensor):
        return self.norm(input + self.dropout(weight))

class PositionWiseFFN(nn.Module):

    def __init__(self, input_size: int, hidden_size: int, output_size: int, dropout_p: float):
        self.linear1 = nn.Linear(input_size, hidden_size)
        self.linear2 = nn.Linear(hidden_size, output_size)
        self.nonlinear = nn.ReLU()
    
    def forward(self, X):
        return self.linear2(self.nonlinear(self.linear1(X)))

class TransformerEncoder(nn.Module):

    def __init__(self, nblock: int, nhead: int, input_size, seq_size, hidden_size, ffn_hidden_size, dropout_p: float):
        super().__init__()
        self.emb = nn.Embedding(input_size, hidden_size)
        self.position_encoding = PositionEncoding(seq_size, hidden_size, dropout_p)
        self.encoder_layer = nn.TransformerEncoderLayer(
            d_model=hidden_size,
            nhead=nhead,
            dim_feedforward=ffn_hidden_size,
            dropout=dropout_p,
            batch_first=True)
        self.encoder = nn.TransformerEncoder(self.encoder_layer, num_layers=nblock)
        self.seq_size = seq_size
        self.hidden_size = hidden_size
    
    def forward(self, input, src_key_padding_mask=None):
        emb = self.position_encoding(self.emb(input) / math.sqrt(self.hidden_size))
        return self.encoder(emb, src_key_padding_mask=src_key_padding_mask)

class TransformerDecoder(nn.Module):

    def __init__(self, nblock, nhead, output_size, seq_size, hidden_size, ffn_hidden_size, dropout_p):
        super().__init__()

        self.embedding = nn.Embedding(output_size, hidden_size)
        self.position_encoding = PositionEncoding(seq_size, hidden_size, dropout_p)
        self.decoder_layer = nn.TransformerDecoderLayer(
            d_model=hidden_size,
            nhead=nhead,
            dim_feedforward=ffn_hidden_size,
            dropout=dropout_p,
            batch_first=True
        )
        self.decoder = nn.TransformerDecoder(
            decoder_layer=self.decoder_layer,
            num_layers=nblock
        )
        self.final_fc = nn.Linear(hidden_size, output_size)
    
    def forward(self, tgt_tensor, memory, tgt_mask, tgt_key_padding_mask=None, memory_key_padding_mask=None):
        tgt_tensor = self.position_encoding(self.embedding(tgt_tensor))
        decoder_output = self.decoder(
            tgt=tgt_tensor,
            memory=memory,
            tgt_mask=tgt_mask,
            tgt_key_padding_mask=tgt_key_padding_mask,
            memory_key_padding_mask=memory_key_padding_mask
        )

        return self.final_fc(decoder_output)

def subsequent_mask(size, device):
    return torch.triu(torch.ones(size, size, device=device), diagonal=1).bool()


class Transformer(nn.Module):

    def __init__(self, encoder, decoder):
        super().__init__()
        self.encoder = encoder
        self.decoder = decoder
    
    def forward(self,
                src_tensor: torch.Tensor,
                decoder_input: torch.Tensor,
                src_key_padding_mask=None,
                tgt_key_padding_mask=None):
        
        seq_size = decoder_input.size(1)
        device = src_tensor.device

        encoder_output = self.encoder(
            input=src_tensor,
            src_key_padding_mask=src_key_padding_mask
        )

        decoder_output = self.decoder(
            tgt_tensor=decoder_input,
            memory=encoder_output,
            tgt_mask=subsequent_mask(seq_size, device),
            tgt_key_padding_mask=tgt_key_padding_mask,
            memory_key_padding_mask=src_key_padding_mask
        )
        
        return decoder_output
    
    def forward_autoregressively(self,
                                 src_tensor: torch.Tensor,
                                 bos_token: int,
                                 eos_token: int,
                                 pad_token: int,
                                 sequence_length: int,
                                 src_key_padding_mask=None):
        
        batch_size = src_tensor.size(0)
        decoder_input = torch.full(
            size=(batch_size, 1),
            fill_value=bos_token,
            dtype=torch.long,
            device=src_tensor.device
        )

        finished = torch.zeros(batch_size, dtype=torch.bool, device=src_tensor.device)
        logits = []

        for _ in range(sequence_length):
            tgt_key_padding_mask = decoder_input == pad_token

            logit:torch.Tensor = self(
                src_tensor=src_tensor,
                decoder_input=decoder_input,
                src_key_padding_mask=src_key_padding_mask,
                tgt_key_padding_mask=tgt_key_padding_mask
            )

            next_token = logit[:, -1, :].argmax(dim=-1, keepdim=True)

            # Mask next_token to pad for finished sequences
            next_token = next_token.masked_fill(finished.unsqueeze(1), pad_token)

            # Update finished flags
            finished = finished | (next_token.squeeze(1) == eos_token)

            decoder_input = torch.cat((decoder_input, next_token), dim=1)
            logits.append(logit[:, -1, :])

            if finished.all():
                break
        
        logits = torch.stack(logits, dim=1)
        emb_size = logits.size(-1)

        if logits.size(1) < sequence_length:
            seq_to_pad = sequence_length - logits.size(1)
            # Need to create a tensor with shape (batch_size, to_pad, emb_size)
            to_pad = torch.full(
                size=(batch_size, seq_to_pad, emb_size),
                fill_value=-1e9,
                dtype=logits.dtype,
                device=logits.device
            )
            logits = torch.cat((logits, to_pad), dim=1)

        return logits

