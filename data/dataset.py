from datasets import load_dataset, load_from_disk
from pathlib import Path
import pathlib


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


def prepare_datasets(data_local_path: Path):

    if data_local_path.exists():
        return load_from_disk(data_local_path)

    # dataset is splitted into train, valid, and test
    datasets = load_dataset("code_search_net", "python", trust_remote_code=True)
    datasets.filter(lambda ds: _filter_dataset(
        ds=ds,
        min_doc_token=0,
        max_doc_token=256,
        min_code_token=0,
        max_code_token=256
    ))

    # only need func_code_tokens and func_documentation_tokens.
    columns_to_save = [
        'func_code_tokens',
        'func_documentation_tokens'
    ]
    _filter_columns_from_dataset(datasets, columns_to_save)

    datasets.save_to_disk(data_local_path)

    return datasets


if __name__ == '__main__':
    data_local_path = pathlib.Path.cwd() / 'data' / 'preprocessed_dataset'
    datasets = prepare_datasets(data_local_path)
    print(datasets)






