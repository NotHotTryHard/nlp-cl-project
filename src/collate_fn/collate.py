import logging
import numpy as np
import torch
from torch.nn.utils.rnn import pad_sequence
import transformers
from transformers import GPT2Tokenizer, GPT2TokenizerFast

try:
    from transformers.data.data_collator import FlaxDataCollatorForT5MLM
except ImportError:
    # This is for compatibility with newer versions of Transformers
    from transformers.data import DataCollatorForLanguageModeling as FlaxDataCollatorForT5MLM

try:
    from src.utils.data_utils import compute_input_and_target_lengths, group_texts
except ImportError:
    # Define dummy functions if imports fail
    def compute_input_and_target_lengths(inputs_length, noise_density, mean_noise_span_length):
        return inputs_length, inputs_length
    def group_texts(texts, block_size):
        return texts

logger = logging.getLogger(__name__)

class CollateClass:
    def __init__(self, tokenizer, max_length, mlm_items=False, mlm_probability=0.15, mean_span_length=3, decoder_start_token_id=0):
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.mlm_items = mlm_items
        self.mlm_probability = mlm_probability
        self.mean_span_length = mean_span_length
        self.decoder_start_token_id = decoder_start_token_id
        
        # Check tokenizer type for GPT-2 vs T5
        self.is_gpt2_tokenizer = isinstance(tokenizer, transformers.GPT2Tokenizer) or isinstance(tokenizer, transformers.GPT2TokenizerFast)
        
        # Ensure GPT-2 tokenizer has a pad token
        if self.is_gpt2_tokenizer and self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
            logger.info("Set GPT-2 tokenizer pad_token to eos_token")
    
    def _generate_spans_per_item(self, input_ids):
        """
        Generate random spans for masking.
        input_ids must be 1-d
        """

        seq_length = (input_ids != self.tokenizer.pad_token_id).sum()
        num_tokens_to_mask = int(seq_length * self.mlm_probability)

        # Use a geometric distribution to decide span lengths
        spans = []
        while num_tokens_to_mask > 0:
            span_len = min(np.random.geometric(p=1 / self.mean_span_length), num_tokens_to_mask)
            start_idx = np.random.randint(0, seq_length - span_len)
            spans.append((start_idx, start_idx + span_len))
            num_tokens_to_mask -= span_len
        
        return spans
    
    def _apply_span_masking_per_item(self, input_ids, spans):
        """
        Replace spans in input_ids with sentinel tokens and generate target sequence.
        input-ids must be 1-d
        """
        sentinel_token_id = self.tokenizer.convert_tokens_to_ids("<extra_id_0>")
        input_ids_masked = input_ids.clone()
        
        for start, end in spans:
            input_ids_masked[start:end] = sentinel_token_id
            sentinel_token_id += 1

        return input_ids_masked

    def _legacy_mlm_call(self, dataset_items):
        inputs = [item['text'] for item in dataset_items]

        input_encodings = self.tokenizer.batch_encode_plus(
            [sentence for sentence in inputs],
            padding="longest",
            max_length=self.max_length,
            truncation=True,
            return_tensors="pt"
        )
        input_ids, attention_mask = input_encodings.input_ids, input_encodings.attention_mask

        input_ids_masked = []
        for i, item in enumerate(input_ids):
            spans = self._generate_spans_per_item(item)
            item_masked = self._apply_span_masking_per_item(item, spans)
            input_ids_masked.append(item_masked)

        input_ids_masked = pad_sequence(
            input_ids_masked,
            batch_first=True,
            padding_value=self.tokenizer.pad_token_id
        )

        return {
            "input_ids": input_ids_masked,
            "attention_mask": attention_mask,
            "labels": input_ids,
            "decoder_attention_mask": None
        }

    def _mlm_call(self, dataset_items):
        # Note: MLM is typically for T5, not GPT-2
        if self.is_gpt2_tokenizer:
            logger.warning("MLM is not typically used with GPT-2, falling back to standard processing")
            return self._process_gpt2_items(dataset_items, is_mlm=True)
            
        inputs = [item['text'] for item in dataset_items]
        input_encodings = self.tokenizer(
            [sentence for sentence in inputs],
            return_attention_mask=False
        )
        input_ids = input_encodings.input_ids
        
        expanded_inputs_length, targets_length = compute_input_and_target_lengths(
            inputs_length=self.max_length,
            noise_density=self.mlm_probability,
            mean_noise_span_length=self.mean_span_length
        )

        input_ids = group_texts(input_ids, expanded_inputs_length)
        
        batch = {"input_ids": np.array(input_ids)}
        
        mlm_data_collator = FlaxDataCollatorForT5MLM(
            tokenizer=self.tokenizer,
            noise_density=self.mlm_probability,
            mean_noise_span_length=self.mean_span_length,
            input_length=self.max_length,
            target_length=targets_length,
            pad_token_id=self.tokenizer.pad_token_id,
            decoder_start_token_id=self.decoder_start_token_id
        )
        
        batch = mlm_data_collator(batch)
        for k, v in batch.items():
            batch[k] = torch.tensor(v)

        batch["attention_mask"] = torch.full(batch["input_ids"].shape, True)
        batch["decoder_attention_mask"] = torch.full(batch["labels"].shape, True)
        
        return batch
    
    def _process_gpt2_items(self, dataset_items, is_mlm=False):
        """
        Process items specifically for GPT-2 models.
        
        For GPT-2, we typically concatenate input and target with a separator
        token, as GPT-2 is an autoregressive model that predicts the next tokens.
        
        Args:
            dataset_items: List of (input, target) pairs or {'text': text} items
            is_mlm: Whether this is for MLM-like processing
            
        Returns:
            Dictionary with keys for input_ids, attention_mask, and labels
        """
        # Ensure GPT-2 tokenizer has a pad token
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
            logger.info("Set GPT-2 tokenizer pad_token to eos_token")
            
        if is_mlm:
            # For MLM-like processing with GPT-2
            inputs = [item['text'] for item in dataset_items]
            
            encodings = self.tokenizer.batch_encode_plus(
                inputs,
                padding="longest",
                max_length=self.max_length,
                truncation=True,
                return_tensors="pt"
            )
            
            input_ids = encodings.input_ids
            attention_mask = encodings.attention_mask
            
            # For GPT-2, labels are the same as inputs, but we mask padding tokens
            labels = input_ids.clone()
            labels[attention_mask == 0] = -100  # -100 is ignored in loss
            
            return {
                "input_ids": input_ids,
                "attention_mask": attention_mask,
                "labels": labels
            }
        else:
            # Check if we have a dictionary-style dataset (like OpenWebText) or tuple-style
            if hasattr(dataset_items[0], 'keys') and 'text' in dataset_items[0]:
                # Dictionary style dataset like OpenWebText
                texts = [item['text'] for item in dataset_items]
                
                encodings = self.tokenizer.batch_encode_plus(
                    texts,
                    padding="longest",
                    max_length=self.max_length,
                    truncation=True,
                    return_tensors="pt"
                )
                
                input_ids = encodings.input_ids
                attention_mask = encodings.attention_mask
                
                # For autoregressive training, labels are the same as inputs with padding masked
                labels = input_ids.clone()
                labels[attention_mask == 0] = -100  # -100 is ignored in loss
                
                return {
                    "input_ids": input_ids,
                    "attention_mask": attention_mask,
                    "labels": labels
                }
            else:
                # Normal (input, target) processing for GPT-2
                inputs = [item[0] for item in dataset_items]
                targets = [item[1] for item in dataset_items]
                
                # For GPT-2, we format as: <input><sep><target>
                combined_texts = []
                for i, t in zip(inputs, targets):
                    # Add a special separator if needed
                    combined_texts.append(f"{i} {t}")
                
                encodings = self.tokenizer.batch_encode_plus(
                    combined_texts,
                    padding="longest",
                    max_length=self.max_length,
                    truncation=True,
                    return_tensors="pt"
                )
                
                input_ids = encodings.input_ids
                attention_mask = encodings.attention_mask
                
                # For GPT-2 training, labels are the same as inputs, but we mask padding tokens
                labels = input_ids.clone()
                labels[attention_mask == 0] = -100  # -100 is ignored in loss
                
                return {
                    "input_ids": input_ids,
                    "attention_mask": attention_mask,
                    "labels": labels
                }

    def __call__(self, dataset_items):
        if self.mlm_items:
            return self._mlm_call(dataset_items)

        # Handle differently based on tokenizer type
        if self.is_gpt2_tokenizer:
            return self._process_gpt2_items(dataset_items)

        # Original T5 processing
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
