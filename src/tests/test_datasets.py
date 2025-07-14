import unittest
from src.datasets import CNNDMHuggingFaceDataset, XSUMHuggingFaceDataset
from transformers import AutoTokenizer

class TestDatasets(unittest.TestCase):
    def test_cnndm_dataset(self):
        dataset = CNNDMHuggingFaceDataset(max_samples=10, split='validation')
        self.assertEqual(len(dataset), 10)
        sample = dataset[0]
        self.assertIn('input', sample)
        self.assertIn('target', sample)
        self.assertTrue(sample['input'].startswith('summarize: '))

    def test_xsum_dataset(self):
        dataset = XSUMHuggingFaceDataset(max_samples=10, split='validation')
        self.assertEqual(len(dataset), 10)
        sample = dataset[0]
        self.assertIn('input', sample)
        self.assertIn('target', sample)
        self.assertTrue(sample['input'].startswith('summarize: '))

    def test_cnndm_filter_length(self):
        tokenizer = AutoTokenizer.from_pretrained('t5-base')
        dataset = CNNDMHuggingFaceDataset(
            max_samples=100, 
            split='validation', 
            filter_max_length=True, 
            max_length=256,
            model_name='t5-base'
        )
        self.assertTrue(len(dataset) < 100)
        for sample in dataset:
            self.assertTrue(len(tokenizer.encode(sample['input'])) <= 256)

    def test_xsum_filter_length(self):
        tokenizer = AutoTokenizer.from_pretrained('t5-base')
        dataset = XSUMHuggingFaceDataset(
            max_samples=100, 
            split='validation', 
            filter_max_length=True, 
            max_length=256,
            model_name='t5-base'
        )
        self.assertTrue(len(dataset) < 100)
        for sample in dataset:
            self.assertTrue(len(tokenizer.encode(sample['input'])) <= 256) 