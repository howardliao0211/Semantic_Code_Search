from data.dataset import get_datasets
from data.tokenizer import Tokenizer
from pathlib import Path
from torch.utils.data import DataLoader
from trainers.core import BaseTrainer
from dataclasses import dataclass
import evaluate
import model
import torch
import model.decoder
import model.encoder
import model.seq2seq

bleu = evaluate.load('bleu')

@dataclass
class CodeDocTrainer(BaseTrainer):

    doc_tokenizer: Tokenizer

    def train_loop(self):
        self.model.train()

        train_loss = 0.0
        for batch, (source_tokens, decoder_input, decoder_output) in enumerate(self.train_loader):
            source_tokens = source_tokens.to(self.device)
            decoder_input = decoder_input.to(self.device)
            decoder_output = decoder_output.to(self.device)

            # Include decoder input for teacher forcing.
            predict = self.model(source_tokens, decoder_input)
            loss = self.loss_fn(predict.view(-1, predict.size(-1)), decoder_output.view(-1))

            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()

            train_loss += loss.item()
            if (batch + 1) % 64 == 0:
                print(f'    loss: {loss.item(): .5f} ---- {batch + 1} / {len(self.train_loader)}')
        
        train_loss /= len(self.train_loader)
        return {'Train Loss': train_loss}
    
    def test_loop(self):
        self.model.eval()
        
        test_loss = 0.0
        references = []
        predictions = []

        for source_tokens, _, decoder_output in self.test_loader:
            source_tokens = source_tokens.to(self.device)
            decoder_output = decoder_output.to(self.device)

            with torch.no_grad():
                predict = self.model(source_tokens)
                loss = self.loss_fn(predict.view(-1, predict.size(-1)), decoder_output.view(-1))
                test_loss += loss.item()
            
                predictions.append(
                    self.doc_tokenizer.to_word(torch.argmax(predict, dim=-1).tolist(), skip_special_tokens=True)
                )

                references.append(
                    [
                        self.doc_tokenizer.to_word(decoder_output.tolist(), skip_special_tokens=True)
                    ]
                )

        # Compute test loss
        test_loss /= len(self.test_loader)

        # Compute Bleu score based on the first sequence in the last batch
        results = bleu.compute(predictions=predictions, references=references)

        # Print Message
        print(f'Test Loss: {test_loss:5f}, Bleu: {results['bleu']:.5f}')
        return {'Test Loss': test_loss, 'Bleu': results['bleu']}

def main():

    # Configure sizes
    input_size = 8192
    output_size = 8192
    batch_size = 128
    
    hidden_size = 64
    sequence_length = 256

    # Get datasets
    DATASET_LOCAL_PATH = Path(r'./preprocessed_dataset')
    code_tokenizer = Tokenizer(input_size)
    doc_tokenizer = Tokenizer(output_size)
    train_dataset, test_dataset, validation_dataset = get_datasets(data_local_path=DATASET_LOCAL_PATH,
                                                      code_tokenizer=code_tokenizer,
                                                      doc_tokenizer=doc_tokenizer,
                                                      sequence_length=sequence_length)

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
    decoder = model.decoder.BahdanauAttentionDecoder(hidden_size, decoder_output_size, code_tokenizer.bos_token, code_tokenizer.eos_token).to(device)
    seq2seq = model.seq2seq.Seq2SeqModel(encoder, decoder).to(device)

    print(f'encoder_input_size: {encoder_input_size}')
    print(f'decoder_output_size: {decoder_output_size}')

    # Prepare Trainer
    trainer = CodeDocTrainer(
        name='Attention_Code2Doc_Model',
        model=seq2seq,
        optimizer=torch.optim.Adam(seq2seq.parameters(), lr=0.001),
        loss_fn=torch.nn.NLLLoss(ignore_index=code_tokenizer.pad_token),
        train_loader=train_loader,
        test_loader=test_loader,
        device=device,
        doc_tokenizer=doc_tokenizer
    )

    trainer.fit(
        epochs=20,
        save_check_point = True,
        graph=True
    )

if __name__ == '__main__':
    main()