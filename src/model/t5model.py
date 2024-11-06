# Adapted from: https://github.com/qcwthu/Lifelong-Fewshot-Language-Learning/blob/cf7d17ce7de6a707d929d0542b3d5e639569855f/Summarization/model.py

import os
import pdb
import sys
import torch
import torch.nn as nn
from torch.nn.functional import kl_div

class T5forSummarization(nn.Module):
    def __init__(self, model, tokenizer, max_length):
        super(T5forSummarization, self).__init__()
        self.model = model
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.decoder_start_token_id_use = self.model.config.decoder_start_token_id

    def _step(
            self, input_ids, attention_mask=None, decoder_input_ids=None, labels=None, decoder_attention_mask=None, **kwargs
    ):
        embeddings = self.model.encoder.embed_tokens(input_ids)
        return self.model(
            inputs_embeds=embeddings,
            attention_mask=attention_mask,
            decoder_input_ids=decoder_input_ids,
            decoder_attention_mask=decoder_attention_mask,
            labels=labels,
            output_attentions=True,
            output_hidden_states=True
        )

    def _step_pre(
            self, input_ids, attention_mask=None, decoder_input_ids=None, labels=None, decoder_attention_mask=None
    ):
        embeddings = self.model.encoder.embed_tokens(input_ids)
        return self.model(
            inputs_embeds=embeddings,
            attention_mask=attention_mask,
            decoder_input_ids=decoder_input_ids,
            decoder_attention_mask=decoder_attention_mask,
            labels=labels,
            output_attentions=True,
            output_hidden_states=True
        )

    def forward(self, batch):
        outputs = self._step(**batch)
        return outputs.loss

    def _generative_step(self, batch):
        embedding = self.model.encoder.embed_tokens(batch["input_ids"])
        decoder_input_ids = (
            torch.ones((batch["input_ids"].shape[0], 1), dtype=torch.long, device=batch["input_ids"].device) * self.decoder_start_token_id_use
        )
        generated_ids = self.model.generate(
            inputs_embeds=embedding,
            decoder_input_ids=decoder_input_ids,
            attention_mask=batch["attention_mask"],
            use_cache=True,
            max_length=128,
            num_beams=4,
            repetition_penalty=2.5,
            length_penalty=1.0,
            early_stopping=True
        )

        preds = self.ids_to_clean_text(generated_ids)
        target = self.ids_to_clean_text(batch["labels"])
        input = self.ids_to_clean_text(batch["input_ids"])
        return input, target, preds

    def _generative_samples(self, batch):
        embedding = self.model.encoder.embed_tokens(batch["input_ids"])
        decoder_input_ids = (
            torch.ones((batch["input_ids"].shape[0], 1), dtype=torch.long, device=batch["input_ids"].device) * self.decoder_start_token_id_use
        )

        generated_ids = self.model.generate(
            inputs_embeds=embedding,
            decoder_input_ids=decoder_input_ids,
            attention_mask=batch["attention_mask"],
            use_cache=True,
            max_length=self.max_length,
            repetition_penalty=2.5,
            length_penalty=1.0,
            early_stopping=True,
            do_sample=True,
            top_k = 64,
            num_return_sequences=3
        )

        preds = self.ids_to_clean_text(generated_ids)
        target = self.ids_to_clean_text(batch["labels"])
        input = self.ids_to_clean_text(batch["input_ids"])
        return input, target, preds


    def ids_to_clean_text(self, generated_ids):
        gen_text = self.tokenizer.batch_decode(
            generated_ids, skip_special_tokens=True, clean_up_tokenization_spaces=False
        )
        return self.lmap(str.strip, gen_text)

    def lmap(self, f, x):
        """list(map(f, x))"""
        return list(map(f, x))
