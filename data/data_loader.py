import datasets

def filter_dataset(ds, min_doc_token: int, max_doc_token: int, min_code_token: int, max_code_token: int, language='python') -> bool:
    # Step 1: Only allow python
    if ds['language'] != language:
        return False
    
    # Step 2: Check if the coden token length if > min_code_token
    if len(ds['func_code_tokens']) < min_code_token or \
        len(ds['func_code_tokens'] > max_code_token):
        return False
    
    if len(ds['func_documentation_tokens']) < min_doc_token or \
        len(ds['func_documentation_tokens']) > max_doc_token:
        return False
    
    # Step 3: Check if the func documentation only include ascii code (exclude non-english).
    if ds['func_documentation_string'].isascii() == False:
        return False

    return True

def get_datasets():
    # dataset is splitted into train, valid, and test
    datasets = datasets.load_dataset("code_search_net", "pythong", trust_remote_code=True)
    datasets.filter(lambda ds: filter_dataset(
        ds=ds,
        min_doc_token=0,
        max_doc_token=256,
        min_code_token=0,
        max_code_token=256
    ))
    return datasets


if __name__ == '__main__':
    datasets = get_datasets
    print(datasets)





