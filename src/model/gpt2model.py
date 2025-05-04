import torch
import torch.nn as nn
import copy
import numpy as np
from transformers import GPT2Tokenizer, GPT2LMHeadModel

class GPT2forGeneration(nn.Module):
    """
    Base GPT-2 model class for generation tasks.
    
    This implementation is compatible with the existing trainer pipeline and follows
    the same interface as T5forSummarization, but modified for GPT-2's decoder-only
    architecture.
    """
    def __init__(self, model_name, cache_dir, max_length, output_hidden_states=True):
        super(GPT2forGeneration, self).__init__()
        self.model = GPT2LMHeadModel.from_pretrained(model_name, cache_dir=cache_dir)
        self.tokenizer = GPT2Tokenizer.from_pretrained(model_name, cache_dir=cache_dir)
        
        # GPT-2 doesn't have a padding token by default - set it to EOS token
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
            self.model.config.pad_token_id = self.model.config.eos_token_id
            
        self.max_length = max_length
        self.output_hidden_states = output_hidden_states
        
    def _step(self, input_ids, attention_mask=None, labels=None, **kwargs):
        """
        Perform a forward pass through the model.
        
        Note: GPT-2 doesn't use decoder_input_ids or decoder_attention_mask
        as it's a decoder-only model.
        """
        # For GPT-2, we need to set position IDs correctly for padded sequences
        # to ensure proper attention masking
        position_ids = attention_mask.long().cumsum(-1) - 1
        position_ids.masked_fill_(attention_mask == 0, 0)
        
        return self.model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            position_ids=position_ids,
            labels=labels,
            output_attentions=True,
            output_hidden_states=self.output_hidden_states
        )
    
    def forward(self, batch):
        """
        Forward pass for training.
        
        For GPT-2, we use input_ids as the input and typically shift labels to predict next tokens.
        However, Hugging Face's implementation automatically handles the shift when labels are provided.
        """
        # Process labels to handle padding for loss calculation
        labels = batch["labels"].clone()
        if "decoder_attention_mask" in batch:
            # If provided, use it for masking labels
            decoder_attention_mask = batch["decoder_attention_mask"]
            labels.masked_fill_(decoder_attention_mask == 0, -100)  # -100 is ignored by CrossEntropyLoss
        else:
            # Otherwise use input attention mask
            labels.masked_fill_(batch["attention_mask"] == 0, -100)
            
        outputs = self._step(
            input_ids=batch["input_ids"],
            attention_mask=batch["attention_mask"],
            labels=labels
        )
        
        return outputs.loss
    
    def _generative_step(self, batch):
        """
        Generate text given input context for evaluation.
        
        Unlike T5 which requires separate encoder/decoder processing,
        GPT-2 directly generates continuations from input_ids.
        """
        generated_ids = self.model.generate(
            input_ids=batch["input_ids"],
            attention_mask=batch["attention_mask"],
            max_length=self.max_length,
            num_beams=2,
            repetition_penalty=2.5,
            length_penalty=1.0,
            early_stopping=True,
            pad_token_id=self.tokenizer.pad_token_id
        )
        
        inputs = self.ids_to_clean_text(batch["input_ids"])
        targets = self.ids_to_clean_text(batch["labels"])
        preds = self.ids_to_clean_text(generated_ids)
        
        return inputs, targets, preds
    
    def _generative_samples(self, batch):
        """
        Generate multiple samples with randomness for evaluation.
        """
        generated_ids = self.model.generate(
            input_ids=batch["input_ids"],
            attention_mask=batch["attention_mask"],
            max_length=self.max_length,
            do_sample=True,
            top_k=50,
            top_p=0.95,
            temperature=0.8,
            repetition_penalty=2.5,
            num_return_sequences=3,
            pad_token_id=self.tokenizer.pad_token_id
        )
        
        # Reshape the output if we have multiple sequences per input
        batch_size = batch["input_ids"].shape[0]
        if generated_ids.shape[0] == 3 * batch_size:
            # Reshape to get sets of 3 sequences for each input
            generated_ids = generated_ids.view(batch_size, 3, -1)[:, 0, :]  # Take first sequence
        
        inputs = self.ids_to_clean_text(batch["input_ids"])
        targets = self.ids_to_clean_text(batch["labels"])
        preds = self.ids_to_clean_text(generated_ids)
        
        return inputs, targets, preds
    
    def ids_to_clean_text(self, generated_ids):
        """
        Convert token IDs to text, removing special tokens.
        """
        gen_text = self.tokenizer.batch_decode(
            generated_ids, skip_special_tokens=True, clean_up_tokenization_spaces=True
        )
        return self.lmap(str.strip, gen_text)
    
    def lmap(self, f, x):
        """list(map(f, x))"""
        return list(map(f, x)) 