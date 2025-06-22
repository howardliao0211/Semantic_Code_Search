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

    def __init__(self) -> None:
        self.word2index = {
            EOS_STR: EOS_IDX,
            BOS_STR: BOS_IDX,
            PAD_STR: PAD_IDX,
            UNK_STR: UNK_IDX
        }

        self.index2word = {
            v: k for k, v in self.word2index.items()
        }

        self.num = len(self.word2index)
    
    @property
    def eos(self) -> str:
        return EOS_STR
    
    @property
    def bos(self) -> str:
        return BOS_STR
    
    @property
    def pad(self) -> str:
        return PAD_STR
    
    def to_idx(self, token):
        
        if isinstance(token, list):
            return [self.to_idx(tok) for tok in token]

        if token not in self.word2index:
            self.word2index[token] = self.num
            self.index2word[self.num] = token
            self.num += 1
        
        return self.word2index[token]
    
    def to_word(self, index: int) -> str:

        if isinstance(index, list):
            return ' '.join([self.to_word(idx) for idx in index])

        return self.index2word.get(index, UNK_STR)

    def __len__(self) -> int:
        return self.num
    
    def save_to_disk(self, path: pathlib.Path) -> None:
        with open(path, 'w', encoding='utf-8') as f:
            json.dump(self.word2index, f, ensure_ascii=False, indent=2)

    @staticmethod
    def load_from_disk(path: pathlib.Path) -> 'Tokenizer':
        with open(path, 'r', encoding='utf-8') as f:
            word2index = json.load(f)

        tokenizer = Tokenizer()
        tokenizer.word2index = word2index
        tokenizer.index2word = {int(v): k for k, v in word2index.items()}
        tokenizer.num = len(word2index)

        return tokenizer


if __name__ == '__main__':
    test_sentence = 'I want to go home'
    test_tokens = test_sentence.split()
    tokenizer = Tokenizer()
    token_index = tokenizer.to_idx(test_tokens)
    print(token_index)
    print(tokenizer.to_word(token_index))


    
    
