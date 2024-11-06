import logging
import torch

from torch.nn.utils.rnn import pad_sequence


logger = logging.getLogger(__name__)

class CollateClass:
    def __init__(self, tokenizer, max_length):
        self.tokenizer = tokenizer
        self.max_length = max_length
    
    def __call__(self, dataset_items):
        inputs = [item[0] for item in dataset_items]
        targets = [item[1] for item in dataset_items]

        input_encodings = self.tokenizer.batch_encode_plus(
            [sentence for sentence in inputs],
            padding="longest",
            max_length=self.max_length,
            truncation=True,
            return_tensors="pt"
        )
        input_ids, attention_mask = input_encodings.input_ids, input_encodings.attention_mask

        target_encodings = self.tokenizer.batch_encode_plus(
            [sentence for sentence in targets],
            padding="longest",
            max_length=self.max_length,
            truncation=True,
            return_tensors="pt"
        )
        labels, decoder_attention_mask = target_encodings.input_ids, target_encodings.attention_mask
        labels[labels == self.tokenizer.pad_token_id] = -100
        
        return {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "labels": labels,
            "decoder_attention_mask": decoder_attention_mask
        }
