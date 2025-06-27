from datasets import load_dataset, load_from_disk
from pathlib import Path
from torch.utils.data import Dataset, DataLoader
from typing import Any
from .tokenizer import Tokenizer
import random
import pathlib
import torch

class CodeDocDataset(Dataset):

    def __init__(self, dataset, sequence_length: int, eos_token: int, bos_token: int, pad_token: int) -> None:
        super().__init__()

        # Use int to represent the index. 
        self.dataset = dataset
        self.sequence_length = sequence_length + 1 # plus 1 for bos / eos
        self.eos_token = eos_token
        self.bos_token = bos_token
        self.pad_token = pad_token
    
    def __len__(self) -> int:
        return len(self.dataset)
    
    def __getitem__(self, index):
        source_tokens = self.dataset[index]['func_code_tokens']
        target_tokens = self.dataset[index]['func_documentation_tokens']
        
        source_tokens = self._pad_or_trunc(source_tokens, self.sequence_length)
        target_tokens = self._pad_or_trunc(target_tokens, self.sequence_length)

        encoder_input = source_tokens + [self.eos_token]
        decoder_input = [self.bos_token] + target_tokens
        decoder_output = target_tokens + [self.eos_token]

        encoder_input = torch.tensor(encoder_input, dtype=torch.int32)
        decoder_input = torch.tensor(decoder_input, dtype=torch.int32)
        decoder_output = torch.LongTensor(decoder_output)

        return (encoder_input, decoder_input, decoder_output)
    
    def show_triplets(self, num: int, code_tokenizer: Tokenizer, doc_tokenizer: Tokenizer) -> None:
        for i in range(num):
            index = random.randint(0, len(self) - 1)
            encoder_input, decoder_input, decoder_output = self[index]
            print(f'#{i}')
            print(f'    encoder_input:  {code_tokenizer.to_word(encoder_input.tolist())}')
            print(f'    decoder_input:  {doc_tokenizer.to_word(decoder_input.tolist())}')
            print(f'    decoder_output: {doc_tokenizer.to_word(decoder_output.tolist())}')
        

    def _pad_or_trunc(self, tokens: list[str], sequence_length: int) -> list[str]:
        if len(tokens) > sequence_length:
            return tokens[:sequence_length]
        
        if len(tokens) < sequence_length:
            return tokens + [self.pad_token for _ in range(sequence_length - len(tokens))]
        
        return tokens


def _filter_dataset(ds, min_doc_token: int, max_doc_token: int, min_code_token: int, max_code_token: int, language='python') -> bool:
    # Step 1: Only allow python
    if ds['language'] != language:
        return False
    
    # Step 2: Check if the coden token length if > min_code_token
    if len(ds['func_code_tokens']) < min_code_token or \
        len(ds['func_code_tokens']) > max_code_token:
        return False
    
    if len(ds['func_documentation_tokens']) < min_doc_token or \
        len(ds['func_documentation_tokens']) > max_doc_token:
        return False
    
    # Step 3: Check if the func documentation only include ascii code (exclude non-english).
    if ds['func_documentation_string'].isascii() == False:
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
    for code_token, doc_token in zip(ds['func_code_tokens'], ds['func_documentation_tokens']):
        if code_token not in allow_code_tokens:
            return False

        if doc_token not in allow_doc_tokens:
            return False

    return True

def _tokenize(example, code_tokenizer: Tokenizer, doc_tokenizer: Tokenizer) -> int:
    return {
        'func_code_tokens': code_tokenizer.to_idx(example['func_code_tokens']),
        'func_documentation_tokens': doc_tokenizer.to_idx(example['func_documentation_tokens']),
    }

def _prepare_datasets_and_tokenizers(data_local_path: Path, code_tokenizer: Tokenizer, doc_tokenizer: Tokenizer, min_doc_token, max_doc_token, min_code_token, max_code_token):

    CODE_TOKENIZER_JSON_STR = 'code_tokenizer_json.json'
    DOC_TOKENIZER_JSON_STR = 'doc_tokenizer_json.json'

    if data_local_path.exists():
        print(f'Loading preprocessed dataset...')
        code_tokenizer.load_from_disk(data_local_path / CODE_TOKENIZER_JSON_STR)
        doc_tokenizer.load_from_disk(data_local_path / DOC_TOKENIZER_JSON_STR)
        return load_from_disk(data_local_path)

    # dataset is splitted into train, valid, and test
    print('Filtering dataset with token size...')
    datasets = load_dataset("code_search_net", "python", trust_remote_code=True)
    datasets = datasets.filter(lambda ds: _filter_dataset(
        ds=ds,
        min_doc_token=min_doc_token,
        max_doc_token=max_doc_token,
        min_code_token=min_code_token,
        max_code_token=max_code_token
    ))

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
                                       allow_doc_tokens=doc_tokenizer.most_freq_tokens)
    )

    # Tokenize datasets
    print('Tokenizing dataset...')
    datasets = datasets.map(lambda example: _tokenize(example, code_tokenizer, doc_tokenizer))

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
        max_doc_token=sequence_length,
        min_code_token=0,
        max_code_token=sequence_length
    )
    train_dataset = CodeDocDataset(datasets['train'], sequence_length, eos_token, bos_token, pad_token)
    test_dataset = CodeDocDataset(datasets['test'], sequence_length, eos_token, bos_token, pad_token)
    validation_dataset = CodeDocDataset(datasets['validation'], sequence_length, eos_token, bos_token, pad_token)

    print(f'Train Dataset Size: {len(train_dataset)}')
    print(f'Test Dataset Size: {len(test_dataset)}')
    print(f'Validation Dataset Size: {len(validation_dataset)}')

    return train_dataset, test_dataset, validation_dataset

if __name__ == '__main__':
    min_doc_token, max_doc_token, min_code_token, max_code_token = 0, 256, 0, 256
    data_local_path = pathlib.Path.cwd() / 'data' / 'preprocessed_dataset'
    tokenizer = Tokenizer()
    train_dataset, test_dataset, validation_dataset = get_datasets(data_local_path, tokenizer, sequence_length=256)
    print(len(tokenizer))








