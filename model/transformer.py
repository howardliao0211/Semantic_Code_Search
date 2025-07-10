from torch import nn
import torch.nn.functional as F
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
        input = input + self.position_emb_matrix[:, :seq_size, :].requires_grad_(False).to(input.device)
        return self.dropout(input)

class Embeddings(nn.Module):

    def __init__(self, input_size, emb_size):
        super().__init__()
        self.emb = nn.Embedding(input_size, emb_size)
        self.emb_size = emb_size
    
    def forward(self, x):
        return self.emb(x) * math.sqrt(self.emb_size)

class TransformerEncoder(nn.Module):

    def __init__(self, nblock: int, nhead: int, input_size, seq_size, hidden_size, ffn_hidden_size, dropout_p: float):
        super().__init__()
        self.emb = Embeddings(input_size, hidden_size)
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
        emb = self.position_encoding(self.emb(input))
        return self.encoder(emb, src_key_padding_mask=src_key_padding_mask)

class TransformerDecoder(nn.Module):

    def __init__(self, nblock, nhead, output_size, seq_size, hidden_size, ffn_hidden_size, dropout_p):
        super().__init__()

        self.embedding = Embeddings(output_size, hidden_size)
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
            memory_key_padding_mask=memory_key_padding_mask,
            tgt_is_causal=True
        )

        return self.final_fc(decoder_output)


class Transformer(nn.Module):

    def __init__(self, encoder, decoder):
        super().__init__()
        self.encoder = encoder
        self.decoder = decoder
        
        for p in self.parameters():
            if p.dim() > 1:
                torch.nn.init.xavier_uniform_(p)

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
            tgt_mask=self._subsequent_mask(seq_size, device),
            tgt_key_padding_mask=tgt_key_padding_mask,
            memory_key_padding_mask=src_key_padding_mask
        )
        
        return decoder_output
    
    def forward_autoregressively(self,
                                 src_tensor: torch.Tensor,
                                 bos_token: int,
                                 eos_token: int,
                                 pad_token: int,
                                 max_len: int,
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

        encoder_output = self.encoder(
            input=src_tensor,
            src_key_padding_mask=src_key_padding_mask
        )

        for _ in range(max_len):
            tgt_key_padding_mask = decoder_input == pad_token

            logit:torch.Tensor = self.decoder(
                tgt_tensor=decoder_input,
                memory=encoder_output,
                tgt_mask=self._subsequent_mask(decoder_input.size(1), src_tensor.device),
                tgt_key_padding_mask=tgt_key_padding_mask,
                memory_key_padding_mask=src_key_padding_mask,
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

        if logits.size(1) < max_len:
            seq_to_pad = max_len - logits.size(1)
            # Need to create a tensor with shape (batch_size, to_pad, emb_size)
            to_pad = torch.full(
                size=(batch_size, seq_to_pad, emb_size),
                fill_value=-1e9,
                dtype=logits.dtype,
                device=logits.device
            )
            logits = torch.cat((logits, to_pad), dim=1)

        return logits
    
    def _subsequent_mask(self, size, device):
        return nn.Transformer.generate_square_subsequent_mask(size, device, dtype=bool)

