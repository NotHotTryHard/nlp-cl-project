import torch
import torch.nn as nn
from transformers import GPT2LMHeadModel, GPT2Tokenizer

class GPT2forGeneration(nn.Module):
    def __init__(self, model_name, cache_dir, max_length, output_hidden_states=True):
        super(GPT2forGeneration, self).__init__()
        # model_name can be "gpt2" for gpt2-small, or "gpt2-medium"
        self.model = GPT2LMHeadModel.from_pretrained(model_name, cache_dir=cache_dir)
        self.tokenizer = GPT2Tokenizer.from_pretrained(model_name, cache_dir=cache_dir)
        
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        self.model.config.pad_token_id = self.tokenizer.pad_token_id

        self.max_length = max_length
        self.output_hidden_states = output_hidden_states

    def forward(self, batch):
        input_ids = batch['input_ids']
        attention_mask = batch['attention_mask']
        labels = batch['labels']
        # GPT2 doesn't use decoder_attention_mask, but it might be in the batch from collate_fn
        decoder_attention_mask = batch.get('decoder_attention_mask')

        # Check if this is language modeling (input_ids and labels are the same)
        # This happens with gpt2_lm_mode=True for datasets like C4
        if torch.equal(input_ids, labels):
            # Pure language modeling - use the text directly
            # GPT2LMHeadModel will handle label shifting internally
            labels_for_lm = labels.clone()
            labels_for_lm[labels_for_lm == self.tokenizer.pad_token_id] = -100
            
            outputs = self.model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                labels=labels_for_lm,
                output_hidden_states=self.output_hidden_states
            )
            return outputs.loss
        
        # Seq2seq mode - concatenate input and target
        if decoder_attention_mask is None and labels is not None:
            decoder_attention_mask = (labels != self.tokenizer.pad_token_id).long()

        # For seq2seq with a decoder-only model, we concatenate input and target
        # and compute loss only on the target part.
        
        # Concatenate input_ids and labels for decoder-only model
        # The model should learn to predict the labels part given the input_ids part.
        
        # Create combined input_ids
        # This assumes that padding is on the right for both input_ids and labels
        combined_input_ids = torch.cat([input_ids, labels], dim=1)
        
        # Create combined attention mask
        if decoder_attention_mask is not None:
            combined_attention_mask = torch.cat([attention_mask, decoder_attention_mask], dim=1)
        else:
            combined_attention_mask = torch.cat([attention_mask, torch.ones_like(labels)], dim=1)


        # Create combined labels for loss calculation
        # We want to ignore the loss for the input part by setting their labels to -100
        input_labels = torch.full_like(input_ids, -100)
        combined_labels = torch.cat([input_labels, labels], dim=1)

        # The model will shift labels internally.
        # We need to make sure padding in the original labels is also ignored.
        combined_labels[combined_labels == self.tokenizer.pad_token_id] = -100

        # Truncate if the combined length is greater than max_length
        if combined_input_ids.shape[1] > self.max_length:
            combined_input_ids = combined_input_ids[:, :self.max_length]
            combined_attention_mask = combined_attention_mask[:, :self.max_length]
            combined_labels = combined_labels[:, :self.max_length]

        outputs = self.model(
            input_ids=combined_input_ids,
            attention_mask=combined_attention_mask,
            labels=combined_labels,
            output_hidden_states=self.output_hidden_states
        )
        return outputs.loss

    def _generative_step(self, batch):
        # For generation, we only use the input_ids as prompt
        generated_ids = self.model.generate(
            input_ids=batch["input_ids"],
            attention_mask=batch["attention_mask"],
            use_cache=True,
            max_length=self.max_length,
            num_beams=2,
            repetition_penalty=2.5,
            length_penalty=1.0,
            early_stopping=True,
            pad_token_id=self.tokenizer.pad_token_id,
            eos_token_id=self.tokenizer.eos_token_id
        )

        # The generated output includes the input prompt. We need to remove it.
        preds = self.ids_to_clean_text(generated_ids[:, batch["input_ids"].shape[1]:])
        target = self.ids_to_clean_text(batch["labels"])
        input_text = self.ids_to_clean_text(batch["input_ids"])
        return input_text, target, preds

    def _generative_samples(self, batch):
        generated_ids = self.model.generate(
            input_ids=batch["input_ids"],
            attention_mask=batch["attention_mask"],
            use_cache=True,
            max_length=self.max_length,
            repetition_penalty=2.5,
            length_penalty=1.0,
            early_stopping=True,
            do_sample=True,
            top_k=64,
            num_return_sequences=3,
            pad_token_id=self.tokenizer.pad_token_id,
            eos_token_id=self.tokenizer.eos_token_id
        )

        preds = self.ids_to_clean_text(generated_ids[:, batch["input_ids"].shape[1]:])
        target = self.ids_to_clean_text(batch["labels"])
        input_text = self.ids_to_clean_text(batch["input_ids"])
        return input_text, target, preds

    def ids_to_clean_text(self, generated_ids):
        gen_text = self.tokenizer.batch_decode(
            generated_ids, skip_special_tokens=True, clean_up_tokenization_spaces=True
        )
        return self.lmap(str.strip, gen_text)

    def lmap(self, f, x):
        """list(map(f, x))"""
        return list(map(f, x)) 