import unittest
from pathlib import Path
from data.dataset import get_datasets, CodeDocDataset
from data.tokenizer import Tokenizer

class DatasetTest(unittest.TestCase):
    
    def __init__(self, methodName = "runTest"):
        super().__init__(methodName)

        # Get datasets
        input_size = 8192
        output_size = 8192
        self.sequence_length = 128

        DATASET_LOCAL_PATH = Path(r'./preprocessed_dataset')
        self.code_tokenizer = Tokenizer(input_size)
        self.doc_tokenizer = Tokenizer(output_size)
        self.train_dataset, self.test_dataset, self.validation_dataset = get_datasets(data_local_path=DATASET_LOCAL_PATH,
                                                                         code_tokenizer=self.code_tokenizer,
                                                                         doc_tokenizer=self.doc_tokenizer,
                                                                         sequence_length=self.sequence_length)

    def test_no_unknown_token(self):
        unk_token = self.code_tokenizer.unk_token

        datasets = [
            self.train_dataset,
            self.test_dataset,
            self.validation_dataset
        ]

        for dataset in datasets:
            for encoder_input, decoder_input, decoder_output in dataset:
                self.assertFalse(
                    unk_token in encoder_input, "Unknown token in encoder_input"
                )

                self.assertFalse(
                    unk_token in decoder_input, "Unknown token in decoder_input"
                )

                self.assertFalse(
                    unk_token in decoder_output, "Unknown token in decoder_output"
                )
    
    def test_include_eos_and_bos(self):
        eos_token, bos_token = self.code_tokenizer.eos_token, self.code_tokenizer.bos_token

        datasets = [
            self.train_dataset,
            self.test_dataset,
            self.validation_dataset
        ]

        for dataset in datasets:
            for encoder_input, decoder_input, decoder_output in dataset:
                self.assertEqual(
                    encoder_input.tolist().count(eos_token), 1, f"Number of eos token in encoder_input: {encoder_input.tolist().count(eos_token)}"
                )

                self.assertEqual(
                    decoder_input.tolist().count(bos_token), 1, f"Number of bos token in decoder_input: {decoder_input.tolist().count(eos_token)}"
                )

                self.assertEqual(
                    decoder_output.tolist().count(eos_token), 1, f"Number of eos token in decoder_output: {decoder_output.tolist().count(eos_token)}"
                )
    
    def test_sequence_length(self):
        datasets = [
            self.train_dataset,
            self.test_dataset,
            self.validation_dataset
        ]

        for dataset in datasets:
            for encoder_input, decoder_input, decoder_output in dataset:
                self.assertEqual(
                    self.sequence_length, len(encoder_input)
                )

                self.assertEqual(
                    self.sequence_length, len(decoder_input)
                )

                self.assertEqual(
                    self.sequence_length, len(decoder_output)
                )

if __name__ == '__main__':
    unittest.main()
