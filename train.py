from data.dataset import get_datasets
from data.tokenizer import Tokenizer
from pathlib import Path
from torch.utils.data import DataLoader
from trainers.core import BaseTrainer
from trainers.utils import graph_loss_animation_start, graph_loss_animation_update, graph_loss_animation_end, load_checkpoint
from dataclasses import dataclass
import datetime
import evaluate
import model
import torch
import model.decoder
import model.encoder
import model.seq2seq
import random
import pathlib

bleu = evaluate.load('bleu')

@dataclass
class CodeDocTrainer(BaseTrainer):

    doc_tokenizer: Tokenizer

    def fit(self, epochs: int, trained_epochs: int=0, graph: bool=False, save_check_point: bool=False) -> None:
        """
        Train the model and optionally plot loss in real-time.
        
        Args:
            epochs (int): Number of training epochs.
        """

        if graph:
            fig, ax, lines, x_data, y_data = None, None, None, None, None

        print("Training the model...")
        for epoch in range(epochs):
            epoch_idx = epoch + trained_epochs + 1

            print(f'============ Epoch {epoch_idx}/{epochs + trained_epochs} ============')

            teacher_forcing_ratio = max(0.5 - epoch * 0.02, 0.1)  # or similar schedule
            use_teacher = random.random() < teacher_forcing_ratio

            print(f'Use Teacher Forcing: {use_teacher}')
            train_state = self.train_loop(use_teacher_forcing=use_teacher)
            test_state = self.test_loop()
            current_statistic = {}
            current_statistic.update(train_state)
            current_statistic.update(test_state)

            if save_check_point:
                # Create Checkpoint Directory
                date = datetime.datetime.today().strftime("%Y%m%d")
                time = datetime.datetime.now().strftime("%H%M%S")
                checkpoint_dir = pathlib.Path(f'Checkpoints') / self.name / date
                checkpoint_dir.mkdir(parents=True, exist_ok=True)
                
                # Create Checkpoint Path
                checkpoint_name = f'{self.name}_epoch{epoch_idx}_{date}_{time}.pt'
                checkpoint_path = str(checkpoint_dir/checkpoint_name)
                checkpoint_dict = self.get_checkpoint_dict(self.model, self.optimizer, epoch_idx, current_statistic)
                torch.save(checkpoint_dict, checkpoint_path)

            if graph:
                if epoch == 0:
                    fig, ax, lines, x_data, y_data = graph_loss_animation_start(
                        stat_names = list(current_statistic.keys()),
                        title=f'{self.name}'
                    )

                graph_loss_animation_update(epoch, current_statistic, ax, lines, x_data, y_data)
            
        if graph:
            graph_loss_animation_end()

    def train_loop(self, use_teacher_forcing: bool):
        self.model.train()

        train_loss = 0.0
        for batch, (source_tokens, decoder_input, decoder_output) in enumerate(self.train_loader):
            source_tokens = source_tokens.to(self.device)
            decoder_input = decoder_input.to(self.device)
            decoder_output = decoder_output.to(self.device)

            # Include decoder input for teacher forcing.
            if use_teacher_forcing:
                predict = self.model(source_tokens, decoder_input)
            else:
                predict = self.model(source_tokens)

            loss = self.loss_fn(predict.view(-1, predict.size(-1)), decoder_output.view(-1))

            self.optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
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
            
                # Convert predicted token IDs to words
                batch_pred_ids = torch.argmax(predict, dim=-1).tolist()  # shape: (batch_size, seq_len)
                batch_preds = self.doc_tokenizer.to_word_batch(batch_pred_ids)  # List[str]
                predictions.extend(batch_preds)  # flat list

                # Convert reference token IDs to words
                batch_refs = self.doc_tokenizer.to_word_batch(decoder_output.tolist())  # List[str]
                references.extend([[ref] for ref in batch_refs])  # wrap each in a list: List[List[str]]

        # Compute test loss
        test_loss /= len(self.test_loader)

        # Compute Bleu score based on the first sequence in the last batch
        results = bleu.compute(predictions=predictions, references=references)

        # Print Message
        print(f'Test Loss: {test_loss:5f}, Bleu: {results['bleu']:.5f}')
        
        # Randomly display one prediction/reference pair
        rand_idx = random.randint(0, len(predictions) - 1)
        print(f"Prediction: {predictions[rand_idx]}")
        print(f"Reference : {references[rand_idx][0]}")
        return {'Test Loss': test_loss, 'Bleu': results['bleu']}

def main():

    # Configure hyperparameters
    input_size = 8192
    output_size = 8192
    batch_size = 64
    hidden_size = 256
    sequence_length = 128
    dropout_p = 0.3
    weight_decay = 1e-5
    learning_rate = 5e-4
    label_smoothing = 0.1

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

    train_dataset.show_triplets(3, code_tokenizer, doc_tokenizer)

    # Prepare model
    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
    encoder_input_size = len(code_tokenizer)
    decoder_output_size = len(doc_tokenizer)
    encoder = model.encoder.RNNEncoder(encoder_input_size, hidden_size, dropout_p=dropout_p).to(device)
    decoder = model.decoder.BahdanauAttentionDecoder(hidden_size,
                                                     decoder_output_size,
                                                     code_tokenizer.bos_token,
                                                     code_tokenizer.eos_token,
                                                     code_tokenizer.pad_token,
                                                     drop_p=dropout_p).to(device)
    seq2seq = model.seq2seq.Seq2SeqModel(encoder, decoder).to(device)

    print(f'encoder_input_size: {encoder_input_size}')
    print(f'decoder_output_size: {decoder_output_size}')

    # Prepare Trainer
    trainer = CodeDocTrainer(
        name='Attention_Code2Doc_Model',
        model=seq2seq,
        optimizer=torch.optim.Adam(seq2seq.parameters(), lr=learning_rate, weight_decay=weight_decay),
        loss_fn=torch.nn.CrossEntropyLoss(ignore_index=code_tokenizer.pad_token, label_smoothing=label_smoothing),
        train_loader=train_loader,
        test_loader=test_loader,
        device=device,
        doc_tokenizer=doc_tokenizer
    )

    trainer.fit(
        epochs=50,
        save_check_point = True,
        graph=True
    )

if __name__ == '__main__':
    main()