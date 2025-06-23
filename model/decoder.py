from torch import nn
from typing import Any
import torch.nn.functional as F
import torch

class RNNDecoder(nn.Module):

    def __init__(self, hidden_size: int, output_size: int) -> None:
        super(RNNDecoder, self).__init__()

        self.emb = nn.Embedding(output_size, hidden_size)
        self.gru = nn.GRU(hidden_size, hidden_size, batch_first=True)
        self.out = nn.Linear(hidden_size, output_size)

    def forward(self, encoder_output: torch.Tensor, encoder_hidden: torch.Tensor, bos_token: int, tgt_tensor: torch.Tensor|Any=None):
        batch_size, seq_size = encoder_output.size(0), encoder_output.size(1)
        decoder_input = torch.empty(batch_size, 1, dtype=torch.long, device=encoder_output.device).fill_(bos_token)

        decoder_outputs = []
        decoder_hidden = encoder_hidden

        for seq_idx in range(seq_size):
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
                decoder_input = out_idx.squeeze(-1)
                
                # need to detach so that gradient will not explode or vanish.
                decoder_input.detach()
        
        # decoder_outputs will be a list of tensor with shape (batch, 1, output_size)
        decoder_outputs = torch.cat(decoder_outputs, dim=1)
        decoder_outputs = F.log_softmax(decoder_outputs, dim=-1)
        return decoder_outputs, decoder_hidden

    def _forward_step(self, input, hidden) -> tuple[torch.Tensor, torch.Tensor]:
        emb_out = F.relu(self.emb(input))
        gru_out, hidden_state = self.gru(emb_out, hidden)
        lin_out = self.out(gru_out)
        return lin_out, hidden_state


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

