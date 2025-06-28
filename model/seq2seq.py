try:
    from .encoder import RNNEncoder
    from .decoder import RNNDecoder, SequenceDecoder
except ImportError:
    from encoder import RNNEncoder
    from decoder import RNNDecoder, SequenceDecoder
from torch import nn
import torch.nn.functional as F
import torch

class Seq2SeqModel(nn.Module):

    def __init__(self, encoder: RNNEncoder, decoder: SequenceDecoder):
        super(Seq2SeqModel, self).__init__()

        self.encoder = encoder
        self.decoder = decoder
        self.apply(self._init_weights)
    
    def forward(self, source_tokens, decoder_inputs = None) -> torch.Tensor:
        encoder_output, encoder_hidden = self.encoder(source_tokens)
        decoder_output, decoder_hidden, attention_weights = self.decoder(encoder_output, encoder_hidden, decoder_inputs)
        return decoder_output
    
    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            torch.nn.init.xavier_uniform_(m.weight)
            m.bias.data.fill_(0.01)

if __name__ == '__main__':
    input_size, hidden_size = 1024, 64
    output_size = input_size

    encoder = RNNEncoder(input_size, hidden_size)
    decoder = RNNDecoder(hidden_size, output_size, bos_token=1)
    seq2seq = Seq2SeqModel(encoder, decoder)

    batch_size = 64
    seq_size = 128

    intput_tensor = torch.randint(1, input_size, (batch_size, seq_size))
    output = seq2seq(intput_tensor, torch.randint(1, output_size, (batch_size, seq_size)))
    
    print(f'output shape (expect {batch_size}, {seq_size}, {output_size}): {output.shape}')

