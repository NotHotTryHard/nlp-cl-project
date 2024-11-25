import logging
import random
import torch

from torch.nn.utils.rnn import pad_sequence


logger = logging.getLogger(__name__)

class CollateClass:
    def __init__(self, tokenizer, max_length, mlm_items=False, mlm_probability=0.15, mean_span_length=3):
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.mlm_items = mlm_items
        self.mlm_probability = mlm_probability
        self.mean_span_length = mean_span_length
    
    def _generate_spans(self, input_ids):
        """
        Generate random spans for masking.
        """
        seq_length = input_ids.size(0)
        num_tokens_to_mask = int(seq_length * self.mlm_probability)

        # Use a geometric distribution to decide span lengths
        spans = []
        while num_tokens_to_mask > 0:
            span_len = min(random.geometric(p=1 / self.mean_span_length), num_tokens_to_mask)
            start_idx = random.randint(0, seq_length - span_len)
            spans.append((start_idx, start_idx + span_len))
            num_tokens_to_mask -= span_len
        return spans
    
    def _apply_span_masking(self, input_ids, spans):
        """
        Replace spans in input_ids with sentinel tokens and generate target sequence.
        """
        sentinel_token_id = self.tokenizer.convert_tokens_to_ids("<extra_id_0>")
        current_sentinel = 0

        input_ids_with_sentinels = input_ids.clone()
        target_ids = []

        for start, end in spans:
            # Replace span in input_ids with current sentinel token
            input_ids_with_sentinels[start:end] = sentinel_token_id + current_sentinel
            current_sentinel += 1

            # Add masked span to target sequence
            target_ids.append(sentinel_token_id + (current_sentinel - 1))
            target_ids.extend(input_ids[start:end])

        # Append padding token at the end of target sequence
        target_ids.append(self.tokenizer.eos_token_id)

        # Pad or truncate target sequence
        target_ids = target_ids[:self.max_length]
        target_ids = torch.tensor(target_ids, dtype=torch.long)

        return input_ids_with_sentinels, target_ids

    def _mlm_call(self, dataset_items):
        inputs = [item['text'] for item in dataset_items]

        input_encodings = self.tokenizer.batch_encode_plus(
            [sentence for sentence in inputs],
            padding="longest",
            max_length=self.max_length,
            truncation=True,
            return_tensors="pt"
        )
        input_ids, attention_mask = input_encodings.input_ids, input_encodings.attention_mask

        spans = self._generate_spans(input_ids)
        input_ids_with_sentinels, target_ids = self._apply_span_masking(input_ids, spans)

        return {
            "input_ids": input_ids_with_sentinels,
            "attention_mask": attention_mask,
            "labels": target_ids,
            "decoder_attention_mask": None
        }

    def __call__(self, dataset_items):
        if self.mlm_items:
            return self._mlm_call(dataset_items)

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
        # labels[labels == self.tokenizer.pad_token_id] = -100
        
        return {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "labels": labels,
            "decoder_attention_mask": decoder_attention_mask
        }
