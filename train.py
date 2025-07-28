import trainers.utils
from data.dataset import get_cleaned_datasets
from data.tokenizer import Tokenizer
from pathlib import Path
from torch.utils.data import DataLoader
from trainers.core import BaseTrainer
from trainers.utils import AnimatePlotter, load_checkpoint
from dataclasses import dataclass
import math
import datetime
import evaluate
import model
import torch
import model.transformer
import model.decoder
import model.encoder
import model.seq2seq
import random
import pathlib

bleu = evaluate.load('bleu')

@dataclass
class CodeDocTrainer(BaseTrainer):

    model: model.transformer.Transformer
    doc_tokenizer: Tokenizer

    def train_loop_overfit_check(self, to_run: int):
        '''
        Check if the model can overfit for a smaller dataset.
        '''

        self.model.train()

        train_loss = 0.0
        source_tokens, decoder_input, decoder_output = next(iter(self.train_loader))

        for idx in range(to_run):
            source_tokens: torch.Tensor = source_tokens.to(self.device)
            decoder_input: torch.Tensor = decoder_input.to(self.device)
            decoder_output: torch.Tensor = decoder_output.to(self.device)
            
            tgt_key_padding_mask = decoder_input == self.doc_tokenizer.pad_token
            src_key_padding_mask = source_tokens == self.doc_tokenizer.pad_token

            # Include decoder input for teacher forcing.
            # Can always train with teaching forcing because tgt_mask is provided.
            predict:torch.Tensor = self.model(
                src_tensor=source_tokens,
                decoder_input=decoder_input,
                src_key_padding_mask=src_key_padding_mask,
                tgt_key_padding_mask=tgt_key_padding_mask
            )

            loss = self.loss_fn(predict.view(-1, predict.size(-1)), decoder_output.view(-1))

            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()
            self.scheduler.step()

            train_loss += loss.item()
            if (idx + 1) % 100 == 0:
                print(f'    loss: {loss.item(): .5f} ---- {idx + 1} / {to_run}')
        
        bos_token = self.doc_tokenizer.bos_token
        eos_token = self.doc_tokenizer.eos_token
        pad_token = self.doc_tokenizer.pad_token

        with torch.no_grad():
            src_key_padding_mask = source_tokens == pad_token
            predict = self.model.forward_autoregressively(
                src_tensor=source_tokens,
                bos_token=bos_token,
                eos_token=eos_token,
                pad_token=pad_token,
                max_len=decoder_output.size(1),
                src_key_padding_mask=src_key_padding_mask
            )
        
            # Convert predicted token IDs to words
            batch_pred_ids = torch.argmax(predict, dim=-1).tolist()  # shape: (batch_size, seq_len)
            batch_preds = self.doc_tokenizer.to_word_batch(batch_pred_ids)  # List[str]

            # Convert reference token IDs to words
            batch_refs = self.doc_tokenizer.to_word_batch(decoder_output.tolist())  # List[str]
        
        for i, (pred, ref) in enumerate(zip(batch_preds, batch_refs)):
            print(f'#{i+1}:')
            print(f'    pred: {pred}')
            print(f'    ref : {ref}')
            

    def train_loop(self):
        self.model.train()

        train_loss = 0.0
        for batch, (source_tokens, decoder_input, decoder_output) in enumerate(self.train_loader):
            source_tokens: torch.Tensor = source_tokens.to(self.device)
            decoder_input: torch.Tensor = decoder_input.to(self.device)
            decoder_output: torch.Tensor = decoder_output.to(self.device)
            
            tgt_key_padding_mask = decoder_input == self.doc_tokenizer.pad_token
            src_key_padding_mask = source_tokens == self.doc_tokenizer.pad_token

            # Include decoder input for teacher forcing.
            # Can always train with teaching forcing because tgt_mask is provided.
            predict:torch.Tensor = self.model(
                src_tensor=source_tokens,
                decoder_input=decoder_input,
                src_key_padding_mask=src_key_padding_mask,
                tgt_key_padding_mask=tgt_key_padding_mask
            )

            loss = self.loss_fn(predict.view(-1, predict.size(-1)), decoder_output.view(-1))

            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()
            self.scheduler.step()

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

            bos_token = self.doc_tokenizer.bos_token
            eos_token = self.doc_tokenizer.eos_token
            pad_token = self.doc_tokenizer.pad_token

            with torch.no_grad():
                src_key_padding_mask = source_tokens == pad_token
                predict = self.model.forward_autoregressively(
                    src_tensor=source_tokens,
                    bos_token=bos_token,
                    eos_token=eos_token,
                    pad_token=pad_token,
                    max_len=decoder_output.size(1),
                    src_key_padding_mask=src_key_padding_mask
                )
                loss = self.loss_fn(predict.contiguous().view(-1, predict.size(-1)), decoder_output.contiguous().view(-1))
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

def get_class_weight_vector(tokenizer: Tokenizer) -> torch.Tensor:
    num_of_class = len(tokenizer)
    weight_vector = torch.ones(num_of_class, dtype=torch.float)
    
    for word, count in tokenizer.counter.items():
        idx = tokenizer.to_idx(word)
        weight_vector[idx] = 1.0 / math.log(count + math.e)
    
    return weight_vector / weight_vector.sum() # normalize weight

def rate(step, model_size, factor, warmup):
    """
    we have to default the step to 1 for LambdaLR function
    to avoid zero raising to negative power.
    """
    if step == 0:
        step = 1
    return factor * (
        model_size ** (-0.5) * min(step ** (-0.5), step * warmup ** (-1.5))
    )

def main():

    # Dataset hyperparameters
    input_size = 50*1_000_000
    output_size = 50*1_000_000
    batch_size = 32
    sequence_length = 128

    # Model hyperparameters
    nblock = 4
    nhead = 4
    hidden_size = 192 
    ffn_hidden_size = 384

    # Traning hyperparameters
    dropout_p = 0.3
    weight_decay = 1e-2
    label_smoothing = 0.1
    scheduler_warmup = 4000

    # Get datasets
    DATASET_LOCAL_PATH = Path(r'data\CodeSearchNet\python')
    code_tokenizer = Tokenizer(input_size)
    doc_tokenizer = Tokenizer(output_size)
    train_dataset, test_dataset, validation_dataset = get_cleaned_datasets(DATASET_LOCAL_PATH, code_tokenizer, doc_tokenizer, sequence_length)
    # train_dataset.show_triplets(1, code_tokenizer, doc_tokenizer, skip_special_tokens=False)

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

    total_params = sum(p.numel() for p in seq2seq.parameters())
    print(f"Total parameters: {total_params:,}")

    # Get class weight
    class_weight = get_class_weight_vector(doc_tokenizer).to(device)

    # Optimizer and Loss Function
    optimizer = torch.optim.AdamW(seq2seq.parameters(), lr=1.0, betas=(0.9, 0.98), eps=1e-9, weight_decay=weight_decay)
    scheduler = torch.optim.lr_scheduler.LambdaLR(
        optimizer=optimizer,
        lr_lambda=lambda step: rate(
            step=step,
            model_size=hidden_size,
            factor=1,
            warmup=scheduler_warmup
        )
    )
    loss_fn = torch.nn.CrossEntropyLoss(ignore_index=code_tokenizer.pad_token, label_smoothing=label_smoothing, weight=class_weight)

    print(f'encoder_input_size: {encoder_input_size}')
    print(f'decoder_output_size: {decoder_output_size}')

    # Load checkpoint
    # checkpoint_path = r'Checkpoints\Transformer_nblock6_nhead4_hidden128_ffn_512_seq32\20250721\Transformer_nblock6_nhead4_hidden128_ffn_512_seq32_epoch1_20250721_224121.pt'
    # checkpoint = load_checkpoint(
    #     checkpoint_path=checkpoint_path,
    #     model=seq2seq,
    #     optimizer=optimizer,
    #     scheduler=scheduler,
    #     device=device
    # )

    # Prepare Trainer
    model_name = f'Transformer_nblock{nblock}_nhead{nhead}_hidden{hidden_size}_ffn_{ffn_hidden_size}seq{sequence_length}'
    trainer = CodeDocTrainer(
        name=model_name,
        model=seq2seq,
        optimizer=optimizer,
        scheduler=scheduler,
        loss_fn=loss_fn,
        train_loader=train_loader,
        test_loader=test_loader,
        plotter=AnimatePlotter(),
        device=device,
        doc_tokenizer=doc_tokenizer,
    )

    trainer.fit(
        epochs=200,
        # trained_epochs=checkpoint['epoch'],
        save_check_point=True,
        graph=True
    )

    # trainer.train_loop_overfit_check(1000)

if __name__ == '__main__':
    main()