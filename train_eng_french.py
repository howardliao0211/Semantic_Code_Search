from data.dataset import get_datasets
from data.tokenizer import Tokenizer
from pathlib import Path
from torch.utils.data import DataLoader
from trainers.core import BaseTrainer
from dataclasses import dataclass
from data.french_dataset import get_dataloader, SOS_token, EOS_token, Lang, MAX_LENGTH, PAD_token
import evaluate
import model
import model.transformer
import torch
import random

random.seed(42)
bleu = evaluate.load('bleu')

@dataclass
class CodeDocTrainer(BaseTrainer):

    input_lang: Lang
    output_lang: Lang
    trained_epochs:int=0

    def fit(self, epochs, trained_epochs = 0, graph = False, save_check_point = False):
        self.trained_epoch = trained_epochs
        return super().fit(epochs, trained_epochs, graph, save_check_point)

    def train_debug(self, to_test:int):
        self.model.train()

        train_loss = 0.0
        source_tokens, decoder_output = next(iter(self.train_loader))
        # src = random.choice(list(self.output_lang.word2index.keys()))
        # tgt = random.choice(list(self.output_lang.word2index.keys()))
        # source_tokens = torch.tensor([self.output_lang.word2index[s] for s in src.split()]).unsqueeze(0).to(self.device)
        # decoder_output = torch.tensor([self.output_lang.word2index[t] for t in tgt.split()] + [EOS_token]).unsqueeze(0).to(self.device)

        with open('debug.txt', 'w') as f:
            pass

        vocab_size = self.output_lang.n_words
        pad_token = PAD_token

        for test_idx in range(to_test):

            source_tokens = source_tokens.to(self.device)
            decoder_input = torch.cat(
                (torch.full((decoder_output.size(0), 1), SOS_token).to(source_tokens.device), decoder_output[:, :-1]),
                dim=-1
            )
            decoder_output = decoder_output.to(self.device)

            predict = self.model(
                src_tensor=source_tokens,
                decoder_input=decoder_input,
                src_key_padding_mask=source_tokens == pad_token,
                tgt_key_padding_mask=decoder_input == pad_token
            )

            loss = self.loss_fn(predict.view(-1, vocab_size), decoder_output.view(-1))

            self.optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
            self.optimizer.step()

            train_loss += loss.item()
            with open('debug.txt', 'a') as f:
                print(f'loss: {loss.item(): .5f} ---- {test_idx:5d} / {to_test}')
                f.write(f'loss: {loss.item(): .5f} ---- {test_idx:5d} / {to_test}\n')
                f.write(f"     Logits:          {predict[0, 0, :5].data.cpu()}\n")  # First token, first few vocab logits
                f.write(f"     Prediction:      {predict.argmax(-1)[0]}\n", )     # Entire first sample
                f.write(f"     Target:          {decoder_output[0]}\n")
                f.write('\n')
                f.write(f"     Source:          {self.input_lang.to_word(source_tokens[0].tolist())}\n")
                f.write(f"     Input :          {self.output_lang.to_word(decoder_input[0].tolist())}\n")
                f.write(f'     Prediction:      {self.output_lang.to_word(predict.argmax(-1)[0].tolist())}\n')
                f.write(f"     Target:          {self.output_lang.to_word(decoder_output[0].tolist())}\n")
                f.write('\n')
                f.write(f"     Target:          {decoder_output[0]}\n")
                f.write(f"     PAD positions:   {(decoder_output[0] == PAD_token).nonzero()}\n")

    def train_loop(self):
        self.model.train()

        train_loss = 0.0
        for batch, (source_tokens, decoder_output) in enumerate(self.train_loader):
            source_tokens = source_tokens.to(self.device)
            tgt_tensor = torch.cat(
                (torch.full((decoder_output.size(0), 1), SOS_token).to(source_tokens.device), decoder_output[:, :-1]),
                dim=-1
            )
            decoder_output = decoder_output.to(self.device)

            tgt_key_padding_mask = tgt_tensor == PAD_token
            src_key_padding_mask = source_tokens == PAD_token
            # Include decoder input for teacher forcing.
            predict = self.model(
                src_tensor=source_tokens,
                decoder_input=tgt_tensor,
                src_key_padding_mask=src_key_padding_mask,
                tgt_key_padding_mask=tgt_key_padding_mask,
            )
            loss = self.loss_fn(predict.view(-1, predict.size(-1)), decoder_output.view(-1))

            self.optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
            self.optimizer.step()

            train_loss += loss.item()
            if (batch + 1) % 64 == 0:
                print(f'    loss: {loss.item(): .5f} ---- {batch + 1} / {len(self.train_loader)}')
        
        self.trained_epochs += 1
        train_loss /= len(self.train_loader)
        return {'Train Loss': train_loss}
    
    def test_loop(self):

        if self.trained_epochs < 30:
            return {}

        self.model.eval()
        
        test_loss = 0.0
        references = []
        predictions = []

        for source_tokens, decoder_output in self.train_loader:
            source_tokens = source_tokens.to(self.device)
            decoder_output = decoder_output.to(self.device)

            with torch.no_grad():
                src_key_padding_mask = source_tokens == PAD_token
                predict = self.model.forward_autoregressively(
                    src_tensor=source_tokens,
                    bos_token=SOS_token,
                    eos_token=EOS_token,
                    pad_token=PAD_token,
                    max_len=decoder_output.size(1),
                    src_key_padding_mask=src_key_padding_mask
                )
                loss = self.loss_fn(predict.contiguous().view(-1, predict.size(-1)), decoder_output.contiguous().view(-1))
                test_loss += loss.item()
            
                # Convert predicted token IDs to words
                batch_pred_ids = torch.argmax(predict, dim=-1).tolist()  # shape: (batch_size, seq_len)
                batch_preds = self.output_lang.to_word_batch(batch_pred_ids)  # List[str]
                predictions.extend(batch_preds)  # flat list

                # Convert reference token IDs to words
                batch_refs = self.output_lang.to_word_batch(decoder_output.tolist())  # List[str]
                references.extend([[ref] for ref in batch_refs])  # wrap each in a list: List[List[str]]

        # Compute test loss
        test_loss /= len(self.train_loader)

        # Compute Bleu score based on the first sequence in the last batch
        try:
            results = bleu.compute(predictions=predictions, references=references)
        except ZeroDivisionError:
            results = {'bleu': 0.0}

        # Print Message
        print(f'Test Loss: {test_loss:5f}, Bleu: {results['bleu']:.5f}')
        
        # Randomly display one prediction/reference pair
        rand_idx = random.randint(0, len(predictions) - 1)
        print(f"Prediction: {predictions[rand_idx]}")
        print(f"Reference : {references[rand_idx][0]}")
        return {'Test Loss': test_loss, 'Bleu': results['bleu']}

def main():

    # Dataset hyperparameters
    input_size = 100000
    output_size = 8192
    batch_size = 32
    sequence_length = MAX_LENGTH

    # Model hyperparameters
    nblock = 2
    nhead = 2
    hidden_size = 128
    ffn_hidden_size = 256

    # Traning hyperparameters
    dropout_p = 0.1
    learning_rate = 5e-4
    weight_decay = 1e-5
    label_smoothing = 0.1

    # Get datasets
    # DATASET_LOCAL_PATH = Path(r'./preprocessed_dataset')
    # code_tokenizer = Tokenizer(input_size)
    # doc_tokenizer = Tokenizer(output_size)
    # train_dataset, test_dataset, validation_dataset = get_datasets(data_local_path=DATASET_LOCAL_PATH,
    #                                                   code_tokenizer=code_tokenizer,
    #                                                   doc_tokenizer=doc_tokenizer,
    #                                                   sequence_length=sequence_length)

    # Create data loaders
    # train_loader = DataLoader(
    #     dataset=train_dataset,
    #     batch_size=batch_size,
    #     shuffle=True
    # )

    # test_loader = DataLoader(
    #     dataset=test_dataset,
    #     batch_size=batch_size,
    #     shuffle=False
    # )

    # val_loader = DataLoader(
    #     dataset=validation_dataset,
    #     batch_size=batch_size,
    #     shuffle=False
    # )

    input_lang, target_lang, train_loader = get_dataloader(batch_size)

    # Prepare model
    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
    encoder_input_size = input_lang.n_words
    decoder_output_size = target_lang.n_words
    encoder = model.transformer.TransformerEncoder(
        nblock=nblock,
        nhead=nhead,
        input_size=encoder_input_size,
        seq_size=sequence_length,
        hidden_size=hidden_size,
        ffn_hidden_size=ffn_hidden_size,
        dropout_p=dropout_p
    ).to(device)

    decoder = model.transformer.TransformerDecoder(
        nblock=nblock,
        nhead=nhead,
        output_size=decoder_output_size,
        seq_size=sequence_length,
        hidden_size=hidden_size,
        ffn_hidden_size=ffn_hidden_size,
        dropout_p=dropout_p
    ).to(device)

    seq2seq = model.transformer.Transformer(
        encoder=encoder,
        decoder=decoder,
        hidden_size=hidden_size,
        output_size=decoder_output_size
    ).to(device)

    print(f'encoder_input_size: {encoder_input_size}')
    print(f'decoder_output_size: {decoder_output_size}')

    # Prepare Trainer
    trainer = CodeDocTrainer(
        name='Transformer_Eng_French',
        model=seq2seq,
        optimizer=torch.optim.Adam(seq2seq.parameters(), lr=learning_rate),
        loss_fn=torch.nn.CrossEntropyLoss(ignore_index=PAD_token),
        train_loader=train_loader,
        test_loader=None,
        device=device,
        input_lang=input_lang,
        output_lang=target_lang,
    )

    trainer.fit(
        epochs=50,
        save_check_point = False,
        graph=True
    )
    # trainer.train_debug(1000)

if __name__ == '__main__':
    main()