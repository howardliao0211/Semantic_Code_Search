from data.dataset import get_datasets
from data.tokenizer import Tokenizer
from pathlib import Path
from torch.utils.data import DataLoader

def main():
    # Get datasets
    DATASET_LOCAL_PATH = Path(r'./preprocessed_dataset')
    tokenizer = Tokenizer()
    train_dataset, test_dataset, validation_dataset = get_datasets(data_local_path=DATASET_LOCAL_PATH,
                                                      tokenizer=tokenizer,
                                                      sequence_length=256)

    # Create data loaders
    train_loader = DataLoader(
        dataset=train_dataset,
        batch_size=256,
        shuffle=True,
        num_workers=4
    )

    test_loader = DataLoader(
        dataset=test_dataset,
        batch_size=256,
        shuffle=False,
        num_workers=4
    )

    val_loader = DataLoader(
        dataset=validation_dataset,
        batch_size=256,
        shuffle=False,
        num_workers=4
    )

    print(train_dataset[0])


if __name__ == '__main__':
    main()