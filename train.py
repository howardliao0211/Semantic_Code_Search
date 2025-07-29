from data.dataset import get_cleaned_datasets
from data.tokenizer import Tokenizer
from pathlib import Path
from torch.utils.data import DataLoader
from dataclasses import dataclass
from torchmetrics.text import BLEUScore
from lightning.pytorch.callbacks.early_stopping import EarlyStopping
import lightning as L
import math
import datetime
import model
import torch
import model.transformer
import model.decoder
import model.encoder
import model.seq2seq
import random
import pathlib

class ModelWrapper(L.LightningModule):

    def __init__(self, model, loss_fn, optim, scheduler, doc_tokenizer: Tokenizer):
        super().__init__()
        self.model = model
        self.loss_fn = loss_fn
        self.optim = optim
        self.scheduler = scheduler
        self.doc_tokenizer = doc_tokenizer
    
    def training_step(self, batch, batch_idx):

        source_tokens, decoder_input, decoder_output = batch

        tgt_key_padding_mask = decoder_input == self.doc_tokenizer.pad_token
        src_key_padding_mask = source_tokens == self.doc_tokenizer.pad_token

        predict:torch.Tensor = self.model(
                src_tensor=source_tokens,
                decoder_input=decoder_input,
                src_key_padding_mask=src_key_padding_mask,
                tgt_key_padding_mask=tgt_key_padding_mask
            )

        loss = self.loss_fn(predict.view(-1, predict.size(-1)), decoder_output.view(-1))
        self.log('train loss', loss, prog_bar=True)

        self.scheduler.step()

        return loss
    
    def test_step(self, batch, batch_idx):
        references = []
        predictions = []

        source_tokens, _, decoder_output = batch
        
        bos_token = self.doc_tokenizer.bos_token
        eos_token = self.doc_tokenizer.eos_token
        pad_token = self.doc_tokenizer.pad_token
        src_key_padding_mask = source_tokens == pad_token

        predict = self.model.forward_autoregressively(
                    src_tensor=source_tokens,
                    bos_token=bos_token,
                    eos_token=eos_token,
                    pad_token=pad_token,
                    max_len=decoder_output.size(1),
                    src_key_padding_mask=src_key_padding_mask
                )

        loss = self.loss_fn(predict.view(-1, predict.size(-1)), decoder_output.view(-1))

        # Convert predicted token IDs to words
        batch_pred_ids = torch.argmax(predict, dim=-1).tolist()  # shape: (batch_size, seq_len)
        batch_preds = self.doc_tokenizer.to_word_batch(batch_pred_ids)  # List[str]
        predictions.extend(batch_preds)  # flat list

        # Convert reference token IDs to words
        batch_refs = self.doc_tokenizer.to_word_batch(decoder_output.tolist())  # List[str]
        references.extend([[ref] for ref in batch_refs])  # wrap each in a list: List[List[str]]
        
        # Compute Bleu score
        bleu = BLEUScore()
        bleu_score = bleu(predictions, references)

        log_dict = {
            'test loss': loss,
            'bleu': bleu_score
        }

        self.log_dict(log_dict, prog_bar=True)
    
    def validation_step(self, batch, batch_idx):
        references = []
        predictions = []

        source_tokens, _, decoder_output = batch
        
        bos_token = self.doc_tokenizer.bos_token
        eos_token = self.doc_tokenizer.eos_token
        pad_token = self.doc_tokenizer.pad_token
        src_key_padding_mask = source_tokens == pad_token

        predict = self.model.forward_autoregressively(
                    src_tensor=source_tokens,
                    bos_token=bos_token,
                    eos_token=eos_token,
                    pad_token=pad_token,
                    max_len=decoder_output.size(1),
                    src_key_padding_mask=src_key_padding_mask
                )

        loss = self.loss_fn(predict.view(-1, predict.size(-1)), decoder_output.view(-1))

        # Convert predicted token IDs to words
        batch_pred_ids = torch.argmax(predict, dim=-1).tolist()  # shape: (batch_size, seq_len)
        batch_preds = self.doc_tokenizer.to_word_batch(batch_pred_ids)  # List[str]
        predictions.extend(batch_preds)  # flat list

        # Convert reference token IDs to words
        batch_refs = self.doc_tokenizer.to_word_batch(decoder_output.tolist())  # List[str]
        references.extend([[ref] for ref in batch_refs])  # wrap each in a list: List[List[str]]
        
        # Compute Bleu score
        bleu = BLEUScore()
        bleu_score = bleu(predictions, references)

        log_dict = {
            'val loss': loss,
            'bleu': bleu_score
        }

        self.log_dict(log_dict, prog_bar=True)

    def configure_optimizers(self):
        return {
            'optimizer': self.optim,
            'lr_scheduler': self.scheduler
        }

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
    batch_size = 64
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
        shuffle=True,
        num_workers=4,
        persistent_workers=True
    )

    test_loader = DataLoader(
        dataset=test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=4,
        persistent_workers=True
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

    # Prepare Trainer
    early_stopping = EarlyStopping(
        monitor='val loss',
    )

    model_wrapper = ModelWrapper(
        model=seq2seq,
        loss_fn=loss_fn,
        optim=optimizer,
        scheduler=scheduler,
        doc_tokenizer=doc_tokenizer
    )

    trainer = L.Trainer(
        callbacks=[early_stopping],
        accumulate_grad_batches=8
    )

    trainer.fit(
        model=model_wrapper,
        train_dataloaders=train_loader,
        val_dataloaders=test_loader,
    )

if __name__ == '__main__':
    main()