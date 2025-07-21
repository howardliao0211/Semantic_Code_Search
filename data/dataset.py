from datasets import load_dataset, load_from_disk
from pathlib import Path
from torch.utils.data import Dataset, DataLoader
from typing import Any
try:
    from .tokenizer import Tokenizer
except ImportError:
    from tokenizer import Tokenizer

import json
import re
import unicodedata
import random
import pathlib
import torch

class CodeDocDataset(Dataset):

    def __init__(self, dataset, sequence_length: int, eos_token: int, bos_token: int, pad_token: int) -> None:
        super().__init__()

        # Use int to represent the index. 
        self.dataset = dataset
        self.sequence_length = sequence_length
        self.eos_token = eos_token
        self.bos_token = bos_token
        self.pad_token = pad_token
    
    def __len__(self) -> int:
        return len(self.dataset)
    
    def __getitem__(self, index):
        source_tokens = self.dataset[index]['func_code_tokens']
        target_tokens = self.dataset[index]['func_documentation_tokens']

        # reserved one more space for bos/eos
        source_len = len(source_tokens)
        target_len = len(target_tokens)

        encoder_input = self._pad_or_trunc(source_tokens, self.sequence_length - 1)
        decoder_input = self._pad_or_trunc(target_tokens, self.sequence_length - 1)
        decoder_output = self._pad_or_trunc(target_tokens, self.sequence_length - 1)

        encoder_input.insert(source_len, self.eos_token)
        decoder_input.insert(0, self.bos_token)
        decoder_output.insert(target_len, self.eos_token)

        encoder_input = torch.LongTensor(encoder_input)
        decoder_input = torch.LongTensor(decoder_input)
        decoder_output = torch.LongTensor(decoder_output)

        return encoder_input, decoder_input, decoder_output
    
    def show_triplets(self, num: int, code_tokenizer: Tokenizer, doc_tokenizer: Tokenizer, skip_special_tokens: bool) -> None:
        for i in range(num):
            index = random.randint(0, len(self) - 1)
            encoder_input, decoder_input, decoder_output = self[index]
            print(f'#{i}')
            print(f'    encoder_input:  {code_tokenizer.to_word(encoder_input.tolist(), skip_special_tokens)}')
            print(f'    decoder_input:  {doc_tokenizer.to_word(decoder_input.tolist(), skip_special_tokens)}')
            print(f'    decoder_output: {doc_tokenizer.to_word(decoder_output.tolist(), skip_special_tokens)}')

    def _pad_or_trunc(self, tokens: list[str], sequence_length: int) -> list[str]:
        if len(tokens) > sequence_length:
            return tokens[:sequence_length]
        
        if len(tokens) < sequence_length:
            return tokens + [self.pad_token for _ in range(sequence_length - len(tokens))]
        
        return tokens.copy()


def _filter_dataset(ds: dict, min_doc_token: int, max_doc_token: int, min_code_token: int, max_code_token: int, language='python') -> bool:
    # Step 1: Only allow python
    if 'language' in ds and ds['language'] != language:
        return False
    
    # Step 2: Check if the coden token length if > min_code_token
    if len(ds['func_code_tokens']) < min_code_token or \
        len(ds['func_code_tokens']) > max_code_token:
        return False
    
    if len(ds['func_documentation_tokens']) < min_doc_token or \
        len(ds['func_documentation_tokens']) > max_doc_token:
        return False
    
    # Step 3: Check if the func documentation only include ascii code (exclude non-english).
    if 'func_documentation_string' in ds and ds['func_documentation_string'].isascii() == False:
        return False

    return True


def _filter_columns_from_dataset(datasets, columns_to_save: list):

    # Get all the columns names
    dataset_columns_to_remove = {
        dataset: columns for dataset, columns in datasets.column_names.items()
    }

    # Remove columns to save from all the column names
    for dataset in dataset_columns_to_remove:
        for column in columns_to_save:
            if column in dataset_columns_to_remove[dataset]:
                dataset_columns_to_remove[dataset].remove(column)

    # Remove all the columns except columns_to_save.
    for dataset in datasets:
        datasets[dataset] = datasets[dataset].remove_columns(dataset_columns_to_remove[dataset])
    
    return datasets

# filter the dataset to only include allow code tokens and allow documentation tokens
# so that the dataset will not have any unknown token
def _filter_tokens(ds, allow_code_tokens, allow_doc_tokens) -> bool:
    for code_token in ds['func_code_tokens']:
        if code_token not in allow_code_tokens:
            return False

    for doc_token in ds['func_documentation_tokens']:
        if doc_token not in allow_doc_tokens:
            return False

    return True

def _encode(example, code_tokenizer: Tokenizer, doc_tokenizer: Tokenizer) -> dict:
    return {
        'func_code_tokens': code_tokenizer.to_idx(example['func_code_tokens']),
        'func_documentation_tokens': doc_tokenizer.to_idx(example['func_documentation_tokens']),
    }

def _unicode_to_ascii(s):
    return ''.join(
        c for c in unicodedata.normalize('NFD', s)
        if unicodedata.category(c) != 'Mn'
    )

# Lowercase, trim, and remove non-letter characters
def _normalize_string(s):
    s = _unicode_to_ascii(s.lower().strip())
    s = re.sub(r"([.!?])", r" \1", s)
    s = re.sub(r"[^a-zA-Z!?]+", r" ", s)
    return s.strip().split()

def _prepare_datasets_and_tokenizers(data_local_path: Path, code_tokenizer: Tokenizer, doc_tokenizer: Tokenizer, min_doc_token, max_doc_token, min_code_token, max_code_token):

    num_process = 4
    CODE_TOKENIZER_JSON_STR = 'code_tokenizer_json.json'
    DOC_TOKENIZER_JSON_STR = 'doc_tokenizer_json.json'

    if data_local_path.exists():
        print(f'Loading preprocessed dataset...')
        code_tokenizer.load_from_disk(data_local_path / CODE_TOKENIZER_JSON_STR)
        doc_tokenizer.load_from_disk(data_local_path / DOC_TOKENIZER_JSON_STR)
        return load_from_disk(data_local_path)

    datasets = load_dataset("code_search_net", "python", trust_remote_code=True)

    # Normalize the documentation string.
    print(f'Creating function tokens...')
    datasets = datasets.map(
        lambda example: {
            'func_documentation_tokens': _normalize_string(example['func_documentation_string']),
            # 'func_code_tokens': _normalize_string(example['func_code_string'])
        },
        num_proc=num_process
    )

    # dataset is splitted into train, valid, and test
    print('Filtering dataset with token size...')
    datasets = datasets.filter(lambda ds: _filter_dataset(
        ds=ds,
        min_doc_token=min_doc_token,
        max_doc_token=max_doc_token,
        min_code_token=min_code_token,
        max_code_token=max_code_token),
        num_proc=num_process
    )

    # only need func_code_tokens and func_documentation_tokens.
    print('Filtering dataset with columns to save')
    columns_to_save = [
        'func_code_string',
        'func_code_tokens',
        'func_documentation_string',
        'func_documentation_tokens'
    ]
    datasets = _filter_columns_from_dataset(datasets, columns_to_save)

    # Load tokens in tokenizer
    code_tokenizer.load_datasets(datasets, 'func_code_tokens')
    doc_tokenizer.load_datasets(datasets, 'func_documentation_tokens')
    print(f'code tokens: {len(code_tokenizer)}')
    print(f'doc_tokens: {len(doc_tokenizer)}')

    # Filter the dataset to only include data with recognizable tokens
    print('Filtering dataset with allow tokens...')
    datasets = datasets.filter(
        lambda example: _filter_tokens(ds=example,
                                       allow_code_tokens=code_tokenizer.most_freq_tokens,
                                       allow_doc_tokens=doc_tokenizer.most_freq_tokens),
        num_proc=num_process
    )

    # Tokenize datasets
    print('Tokenizing dataset...')
    datasets = datasets.map(
        lambda example: _encode(example, code_tokenizer, doc_tokenizer),
        num_proc=num_process
    )

    # Save datasets and tokenizers to local disk. 
    print('Saving dataset to disk...')
    datasets.save_to_disk(data_local_path)
    code_tokenizer.save_to_disk(data_local_path / CODE_TOKENIZER_JSON_STR)
    doc_tokenizer.save_to_disk(data_local_path / DOC_TOKENIZER_JSON_STR)

    return datasets


def get_datasets(data_local_path: Path, code_tokenizer: Tokenizer, doc_tokenizer: Tokenizer, sequence_length: int):
    eos_token, bos_token, pad_token = code_tokenizer.eos_token, code_tokenizer.bos_token, code_tokenizer.pad_token

    datasets = _prepare_datasets_and_tokenizers(
        data_local_path=data_local_path,
        code_tokenizer=code_tokenizer,
        doc_tokenizer=doc_tokenizer,
        min_doc_token=0,
        max_doc_token=sequence_length - 1, # preserve one space for eos/bos
        min_code_token=0,
        max_code_token=sequence_length - 1 # preserve one space for eos/bos
    )
    train_dataset = CodeDocDataset(datasets['train'], sequence_length, eos_token, bos_token, pad_token)
    test_dataset = CodeDocDataset(datasets['test'], sequence_length, eos_token, bos_token, pad_token)
    validation_dataset = CodeDocDataset(datasets['validation'], sequence_length, eos_token, bos_token, pad_token)

    print(f'Train Dataset Size: {len(train_dataset)}')
    print(f'Test Dataset Size: {len(test_dataset)}')
    print(f'Validation Dataset Size: {len(validation_dataset)}')

    return train_dataset, test_dataset, validation_dataset

def _read_codesearchnet_json(json_file) -> dict:
    dataset = {}

    with open(json_file, encoding='utf-8') as f:
        for idx, line in enumerate(f):
            line=line.strip()
            js=json.loads(line)
            if 'idx' not in js:
                js['idx']=idx
            code=' '.join(js['code_tokens']).replace('\n',' ')
            code=' '.join(code.strip().split())
            nl=' '.join(js['docstring_tokens']).replace('\n','')
            nl=' '.join(nl.strip().split())
            
            dataset[idx] = {
                'func_code_tokens': code.split(),
                'func_documentation_tokens': _normalize_string(nl)
            }
    
    return dataset

def get_cleaned_datasets(data_local_path: Path, code_tokenizer: Tokenizer, doc_tokenizer: Tokenizer, sequence_length: int):

    min_sequence_length = 3

    # load dataset
    print(f'Reading datasets')
    datasets = {}
    datasets['train'] = _read_codesearchnet_json(str(data_local_path/'train.jsonl'))
    datasets['test']  = _read_codesearchnet_json(str(data_local_path/'test.jsonl'))
    datasets['validation']  = _read_codesearchnet_json(str(data_local_path/'valid.jsonl'))

    # Filter by length
    print(f'Filtering dataset by length (word-level)')
    for key in datasets:
        filter_iter = filter_iter = filter(
            lambda x: _filter_dataset(x, min_sequence_length, sequence_length-1, min_sequence_length, sequence_length-1),
            datasets[key].values()
        )
        datasets[key] = {
            idx: value for idx, value in enumerate(filter_iter)
        }

    # Building tokenizer
    code_tokenizer.load_datasets(datasets, 'func_code_tokens')
    doc_tokenizer.load_datasets(datasets, 'func_documentation_tokens')
    print(f'code tokens: {len(code_tokenizer)}')
    print(f'doc_tokens: {len(doc_tokenizer)}')

    # Encode tokens
    print(f'Encoding')
    for key in datasets:
        map_iter = map(
            lambda x: _encode(x, code_tokenizer, doc_tokenizer),
            datasets[key].values()
        )
    
        datasets[key] = {
            idx: value for idx, value in enumerate(map_iter)
        }
    
    # Filter by length
    print(f'Filtering dataset by length (token-level)')
    for key in datasets:
        filter_iter = filter(
            lambda x: _filter_dataset(x, min_sequence_length, sequence_length-1, min_sequence_length, sequence_length-1),
            datasets[key].values()
        )
        datasets[key] = {
            idx: value for idx, value in enumerate(filter_iter)
        }

    eos_token, bos_token, pad_token = code_tokenizer.eos_token, code_tokenizer.bos_token, code_tokenizer.pad_token

    train_dataset = CodeDocDataset(datasets['train'], sequence_length, eos_token, bos_token, pad_token)
    test_dataset = CodeDocDataset(datasets['test'], sequence_length, eos_token, bos_token, pad_token)
    validation_dataset = CodeDocDataset(datasets['validation'], sequence_length, eos_token, bos_token, pad_token)

    print(f'Train Dataset Size: {len(train_dataset)}')
    print(f'Test Dataset Size: {len(test_dataset)}')
    print(f'Validation Dataset Size: {len(validation_dataset)}')

    return train_dataset, test_dataset, validation_dataset

if __name__ == '__main__':
    # Dataset hyperparameters
    input_size = 100000
    output_size = 8192
    batch_size = 32
    sequence_length = 32

    DATASET_LOCAL_PATH = Path(r'data\CodeSearchNet\python')
    code_tokenizer = Tokenizer(input_size)
    doc_tokenizer = Tokenizer(output_size)
    train_dataset, test_dataset, validation_dataset = get_cleaned_datasets(DATASET_LOCAL_PATH, code_tokenizer, doc_tokenizer, sequence_length)
    train_dataset.show_triplets(1, code_tokenizer, doc_tokenizer, skip_special_tokens=False)








