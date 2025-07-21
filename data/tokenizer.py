from collections import Counter
import tokenizers
import pathlib
import json

import tokenizers.normalizers
import tokenizers.pre_tokenizers
import tokenizers.trainers

EOS_IDX = 0
BOS_IDX = 1
PAD_IDX = 2
UNK_IDX = 3

EOS_STR = '<eos>'
BOS_STR = '<bos>'
PAD_STR = '<pad>'
UNK_STR = '<unk>'

class Tokenizer:

    def __init__(self, size: int) -> None:
        self.size = size

        self.backend = tokenizers.Tokenizer(tokenizers.models.BPE(unk_token=UNK_STR))
        self.backend.normalizer = tokenizers.normalizers.Sequence(
            [tokenizers.normalizers.NFD(), tokenizers.normalizers.Lowercase(), tokenizers.normalizers.StripAccents()]
        )
        self.backend.pre_tokenizer = tokenizers.pre_tokenizers.BertPreTokenizer()
    
    @property
    def eos(self) -> str:
        return EOS_STR
    
    @property
    def bos(self) -> str:
        return BOS_STR
    
    @property
    def pad(self) -> str:
        return PAD_STR
    
    @property
    def unk(self) -> str:
        return UNK_STR
    
    @property
    def bos_token(self) -> int:
        return self.backend.token_to_id(BOS_STR)
    
    @property
    def eos_token(self) -> int:
        return self.backend.token_to_id(EOS_STR)
    
    @property
    def pad_token(self) -> int:
        return self.backend.token_to_id(PAD_STR)
    
    @property
    def unk_token(self) -> int:
        return self.backend.token_to_id(UNK_STR)
    
    @property
    def counter(self) -> dict:
        return self.backend.get_vocab()
    
    def load_datasets(self, datasets, key):
        special_tokens = [
            BOS_STR, EOS_STR, PAD_STR, UNK_STR
        ]
        trainer = tokenizers.trainers.BpeTrainer(vocab_size=self.size, special_tokens=special_tokens)
        self.backend.train_from_iterator(
            self._datasets_iterator(datasets, key),
            trainer
        )
        self.most_freq_tokens = set(self.backend.get_vocab())

    def _datasets_iterator(self, datasets:dict, key: str):
        for dataset in datasets.values():
            if isinstance(dataset, dict):
                for data in dataset.values():
                    yield data[key]
            else:
                for data in dataset:
                    yield data[key]

    def to_idx(self, token):

        if isinstance(token, list):
            encoding = self.backend.encode(' '.join(token))
        else:
            encoding = self.backend.encode(token)

        return encoding.ids
    
    def to_word(self, index: int, skip_special_tokens: bool=True) -> str:
        return self.backend.decode(index, skip_special_tokens)
    
    def to_word_batch(self, batched_indices) -> list[str]:
        word_batch = []

        for indices in batched_indices:
            words = []
            words.append(self.to_word(indices, skip_special_tokens=True) + ' ' + EOS_STR)
            word_batch.append(' '.join(words))

        return word_batch

    def __len__(self) -> int:
        return self.backend.get_vocab_size()
    
    def save_to_disk(self, path: pathlib.Path) -> None:
        self.backend.save(str(path))

    def load_from_disk(self, path: pathlib.Path):
        self.backend:tokenizers.Tokenizer = tokenizers.Tokenizer.from_file(str(path))
        self.most_freq_tokens = set(self.backend.get_vocab())

if __name__ == '__main__':
    datasets = {}


    
    
