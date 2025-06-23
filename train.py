from data.dataset import get_datasets
from data.tokenizer import Tokenizer
from pathlib import Path
from torch.utils.data import DataLoader
from trainers.core import BaseTrainer
from dataclasses import dataclass
import model
import torch
import model.decoder
import model.encoder
import model.seq2seq

@dataclass
class CodeDocTrainer(BaseTrainer):

    bos_token: int

    def train_loop(self):
        self.model.train()

        train_loss = 0.0
        for batch, (source_tokens, decoder_input, decoder_output) in enumerate(self.train_loader):
            source_tokens = source_tokens.to(self.device)
            decoder_input = decoder_input.to(self.device)
            decoder_output = decoder_output.to(self.device)

            predict = self.model(source_tokens, self.bos_token, decoder_input)
            loss = self.loss_fn(predict.view(-1, predict.size(-1)), decoder_output.view(-1))

            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()

            train_loss += loss.item()
            if (batch + 1) % 64 == 0:
                index = (batch + 1) * self.train_loader.batch_size
                print(f'    loss: {loss.item()} ---- {index:5f} / {len(self.train_loader.dataset)}')
        
        train_loss /= len(self.train_loader.dataset)
        return {'Train Loss': train_loss}
    
    def test_loop(self):
        self.model.eval()
        
        test_loss = 0.0

        for source_tokens, _, decoder_output in self.test_loader:
            source_tokens = source_tokens.to(self.device)
            decoder_output = decoder_output.to(self.device)

            with torch.no_grad():
                predict = self.model(source_tokens, self.bos_token)
                loss = self.loss_fn(predict.view(-1, predict.size(-1)), decoder_output.view(-1))
                test_loss += loss.item()

        test_loss /= len(self.test_loader.dataset)
        print(f'Test Loss: {test_loss:5f}')
        
        return {'Test Loss': test_loss}

def main():

    # Configure sizes
    input_size = 8192
    output_size = 8192
    batch_size = 64
    hidden_size = 64

    # Get datasets
    DATASET_LOCAL_PATH = Path(r'./preprocessed_dataset')
    code_tokenizer = Tokenizer(input_size)
    doc_tokenizer = Tokenizer(output_size)
    train_dataset, test_dataset, validation_dataset = get_datasets(data_local_path=DATASET_LOCAL_PATH,
                                                      code_tokenizer=code_tokenizer,
                                                      doc_tokenizer=doc_tokenizer,
                                                      sequence_length=256)

    # Create data loaders
    train_loader = DataLoader(
        dataset=train_dataset,
        batch_size=batch_size,
        shuffle=True
    )

    test_loader = DataLoader(
        dataset=test_dataset,
        batch_size=batch_size,
        shuffle=False
    )

    val_loader = DataLoader(
        dataset=validation_dataset,
        batch_size=batch_size,
        shuffle=False
    )

    # Prepare model
    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
    encoder_input_size = len(code_tokenizer)
    decoder_output_size = len(doc_tokenizer)
    encoder = model.encoder.RNNEncoder(encoder_input_size, hidden_size).to(device)
    decoder = model.decoder.RNNDecoder(hidden_size, decoder_output_size).to(device)
    seq2seq = model.seq2seq.Seq2SeqModel(encoder, decoder).to(device)

    print(f'encoder_input_size: {encoder_input_size}')
    print(f'decoder_output_size: {decoder_output_size}')

    # Prepare Trainer
    trainer = CodeDocTrainer(
        name='Code DocString Model',
        model=seq2seq,
        optimizer=torch.optim.Adam(seq2seq.parameters(), lr=0.001),
        loss_fn=torch.nn.NLLLoss(),
        train_loader=train_loader,
        test_loader=test_loader,
        device=device,
        bos_token=code_tokenizer.bos_token
    )

    trainer.fit(
        epochs=20,
        graph=True
    )

if __name__ == '__main__':
    main()