"""
1. Tokenize every words into a number.
'Trains',
  'a',
  'k',
  '-',
  'nearest',
  'neighbors',
  'classifier',
  'for',
  'face',
  'recognition',
  '.'])

2. If the tokens is source, add <eos> at the end.

3. If the token is target, add <bos> at the beginning.
"""
from collections import Counter
import pathlib
import json

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
        self.word2index = {
            EOS_STR: EOS_IDX,
            BOS_STR: BOS_IDX,
            PAD_STR: PAD_IDX,
            UNK_STR: UNK_IDX
        }
        self.index2word = {
            v: k for k, v in self.word2index.items()
        }
        self.counter = Counter()
        self.topk_freq = size - 4 # reserve 4 spaces for special tokens
        self.num = 4
    
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
        return BOS_IDX
    
    @property
    def eos_token(self) -> int:
        return EOS_IDX
    
    @property
    def pad_token(self) -> int:
        return PAD_IDX
    
    @property
    def unk_token(self) -> int:
        return UNK_IDX
    
    def load_datasets(self, datasets, key):
        for dataset in datasets.values():
            for data in dataset:
                self._update_tokens(data[key])
        self._build_vocab()

    def _update_tokens(self, tokens: list[str]):
        self.counter.update(tokens)

    def _build_vocab(self):
        # Build the most freq tokens from the most frequent tokens
        self.most_freq_tokens = {word for word, freq in self.counter.most_common(self.topk_freq)}

        # Build word2index and index2word based on the most frequent tokens
        for token in self.most_freq_tokens:
            self.word2index[token] = self.num
            self.index2word[self.num] = token
            self.num += 1
    
    def to_idx(self, token):
        
        if isinstance(token, list):
            return [self.to_idx(tok) for tok in token]

        return self.word2index.get(token, self.unk_token)
    
    def to_word(self, index: int, skip_special_tokens: bool=True) -> str:

        if isinstance(index, list):
            return ' '.join([self.to_word(idx, skip_special_tokens) for idx in index if self.to_word(idx, skip_special_tokens)])

        if skip_special_tokens:
            if index in (EOS_IDX, BOS_IDX, UNK_IDX, PAD_IDX):
                return ''
        
        return self.index2word.get(index, UNK_STR)
    
    def to_word_batch(self, batched_indices) -> list[str]:
        word_batch = []

        for indices in batched_indices:
            words = []
            for idx in indices:
                if idx == PAD_IDX:  # stop at EOS or PAD
                    continue
                words.append(self.to_word(idx, skip_special_tokens=False))
                if idx == EOS_IDX:
                    break
            word_batch.append(' '.join(words))

        return word_batch

    def __len__(self) -> int:
        return self.num
    
    def save_to_disk(self, path: pathlib.Path) -> None:
        data = {
            'word2index': self.word2index,
            'counter': dict(self.counter),
            'topk_freq': self.topk_freq,
            'num': self.num,
            'most_freq_tokens': list(self.most_freq_tokens)  # Convert set to list for JSON
        }
        with open(path, 'w', encoding='utf-8') as f:
            json.dump(data, f, ensure_ascii=False, indent=2)

    def load_from_disk(self, path: pathlib.Path):
        with open(path, 'r', encoding='utf-8') as f:
            data = json.load(f)

        self.word2index = data['word2index']
        self.index2word = {v: k for k, v in self.word2index.items()}
        self.counter = Counter(data['counter'])
        self.topk_freq = data['topk_freq']
        self.num = data['num']
        self.most_freq_tokens = set(data['most_freq_tokens'])  # Restore set from list


if __name__ == '__main__':
    tokenizer = Tokenizer(size=100)
    tokenizer._update_tokens('I am testing this is as;dlkfj asdfsa e a nonsense e sdfasdf eeee csakjdfsad '.split())
    tokenizer._update_tokens('I am testing this is as;dlkfj asdfsa e e sdfasdf eeee csakjdfsad '.split())
    tokenizer._update_tokens('I am testing this is as;dlkfj asdfsa e e sdfasdf eeee csakjdfsad '.split())
    tokenizer._update_tokens('I am testing this is as;dlkfj asdfsa e e sdfasdf eeee csakjdfsad '.split())
    tokenizer._update_tokens('I am testing this is as;dlkfj asdfsa e e sdfasdf eeee csakjdfsad '.split())
    tokenizer._update_tokens('I am testing this is as;dlkfj asdfsa e e sdfasdf eeee csakjdfsad '.split())
    tokenizer._update_tokens('I am testing this is as;dlkfj asdfsa e e sdfasdf eeee csakjdfsad '.split())
    tokenizer._update_tokens('I am testing this is as;dlkfj asdfsa e e sdfasdf eeee csakjdfsad '.split())
    tokenizer._update_tokens('I am testing this is as;dlkfj asdfsa e e sdfasdf eeee csakjdfsad sentence'.split())
    tokenizer._build_vocab()

    tokens = tokenizer.to_idx('this is a nonsense sentence'.split())
    print(tokenizer.to_word(tokens))

    
    
