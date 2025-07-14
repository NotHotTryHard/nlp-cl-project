# Adapted from here: https://github.com/google-research/multilingual-t5/blob/master/multilingual_t5/evaluation/metrics.py



# First, metrics from t5 library:

# Copyright 2024 The T5 Authors.
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Utilities for Question Answering (QA) evaluation.

Matches results on the SQuAD (v1.1) and TriviaQA (v1.0) evaluation scripts.
"""

import collections
import re
import string

from absl import logging
import numpy as np


def _normalize_answer(text, punc_chars, punc_repl):
  """Lower text and remove punctuation, articles and extra whitespace."""

  def remove_articles(s):
    return re.sub(r"\b(a|an|the)\b", " ", s)

  def replace_punctuation(s):
    to_replace = set(punc_chars)
    return "".join(punc_repl if ch in to_replace else ch for ch in s)

  def white_space_fix(s):
    return " ".join(s.split())

  text = text.lower()
  text = replace_punctuation(text)
  text = remove_articles(text)
  text = white_space_fix(text)

  return text


def normalize_trivia_qa(answer):
  """Normalization used in official TriviaQA evaluation script."""
  return _normalize_answer(
      answer, punc_chars=string.punctuation + "‘’´`_", punc_repl=" ").strip()


def normalize_squad(answer):
  """Normalization used in official SQuAD evaluation script."""
  return _normalize_answer(answer, punc_chars=string.punctuation, punc_repl="")


def _metric_max_over_ground_truths(metric_fn, ground_truths, prediction):
  """Computes the maximum of the metric over all ground truths."""
  return max(
      metric_fn(ground_truth, prediction) for ground_truth in ground_truths
  )


def _exact_match_score(target, prediction):
  return target == prediction


def _f1_score(target, prediction):
  """Computes token f1 score for a single target and prediction."""
  prediction_tokens = prediction.split()
  target_tokens = target.split()
  common = (collections.Counter(prediction_tokens) &
            collections.Counter(target_tokens))
  num_same = sum(common.values())
  if num_same == 0:
    return 0
  precision = 1.0 * num_same / len(prediction_tokens)
  recall = 1.0 * num_same / len(target_tokens)
  f1 = (2 * precision * recall) / (precision + recall)
  return f1


def qa_metrics(targets, predictions):
  """Computes exact match and f1 QA scores, expecting pre-normalized text."""
  if len(targets) != len(predictions):
    raise ValueError("Number of targets and predictions must match.")
  em = np.mean([
      _metric_max_over_ground_truths(_exact_match_score, t, p)
      for p, t in zip(predictions, targets)
  ])
  f1 = np.mean([
      _metric_max_over_ground_truths(_f1_score, t, p)
      for p, t in zip(predictions, targets)
  ])
  em *= 100
  f1 *= 100
  logging.info("EM = %.2f, F1 = %.2f", em, f1)
  return {"em": em, "f1": f1}


def qa_em(targets, predictions):
  if len(targets) != len(predictions):
    raise ValueError("Number of targets and predictions must match.")
  return np.mean([
      _metric_max_over_ground_truths(_exact_match_score, t, p)
      for p, t in zip(predictions, targets)
  ]) * 100

def qa_f1(targets, predictions):
  if len(targets) != len(predictions):
    raise ValueError("Number of targets and predictions must match.")
  return np.mean([
      _metric_max_over_ground_truths(_f1_score, t, p)
      for p, t in zip(predictions, targets)
  ]) * 100
  

# Copyright 2022 The mT5 Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Implementation of various metrics to be used with the T5 library.
"""

import collections
import re
import string
import sys
import unicodedata


def normalize_mlqa(s, lang, punct):
  """Lower text and remove punctuation, articles and extra whitespace.

  Based on third_party/py/xtreme/third_party/evaluate_mlqa.py
  Args:
    s: string, typically the answer span predicted by a QA model.
    lang: ISO code of language.
    punct: set of punctuation characters.

  Returns:
    string, after applying normalization rules.
  """

  whitespace_langs = ['en', 'es', 'hi', 'vi', 'de', 'ar']
  mixed_segmentation_langs = ['zh']

  def whitespace_tokenize(text):
    return text.split()

  def mixed_segmentation(text):
    segs_out = []
    temp_str = ''
    for char in text:
      if re.search(r'[\u4e00-\u9fa5]', char) or char in punct:
        if temp_str != '':
          ss = whitespace_tokenize(temp_str)
          segs_out.extend(ss)
          temp_str = ''
        segs_out.append(char)
      else:
        temp_str += char
    if temp_str != '':
      ss = whitespace_tokenize(temp_str)
      segs_out.extend(ss)
    return segs_out

  def drop_articles(text, lang):
    if lang == 'en':
      return re.sub(r'\b(a|an|the)\b', ' ', text)
    elif lang == 'es':
      return re.sub(r'\b(un|una|unos|unas|el|la|los|las)\b', ' ', text)
    elif lang == 'hi':
      return text
    elif lang == 'vi':
      return re.sub(r'\b(của|là|cái|chiếc|những)\b', ' ', text)
    elif lang == 'de':
      return re.sub(
          r'\b(ein|eine|einen|einem|eines|einer|der|die|das|den|dem|des)\b',
          ' ', text)
    elif lang == 'ar':
      return re.sub('\sال^|ال', ' ', text)
    elif lang == 'zh':
      return text

  def white_space_fix(text, lang):
    if lang in whitespace_langs:
      tokens = whitespace_tokenize(text)
    elif lang in mixed_segmentation_langs:
      tokens = mixed_segmentation(text)
    return ' '.join([t for t in tokens if t.strip()])

  def drop_punc(text):
    return ''.join(c for c in text if c not in punct)

  s = s.lower()
  s = drop_punc(s)
  s = drop_articles(s, lang)
  s = white_space_fix(s, lang)
  return s

def mlqa(targets, predictions, lang=None):
  """Computes MLQA metrics, maximizing over answers per question.

  Args:
    targets: list of lists of strings
    predictions: list of strings
    lang: ISO code of language

  Returns:
    dict with score_key: squad score across all targets and predictions
  """
  assert lang is not None
  punct = {
      chr(i)
      for i in range(sys.maxunicode)
      if unicodedata.category(chr(i)).startswith('P')
  }.union(string.punctuation)
  targets = [[normalize_mlqa(t, lang, punct) for t in u] for u in targets]
  predictions = [normalize_mlqa(p, lang, punct) for p in predictions]
  return qa_metrics(targets, predictions)


def _mlqa_metrics_prepare(targets, predictions, lang=None):
  assert lang is not None
  punct = {
      chr(i)
      for i in range(sys.maxunicode)
      if unicodedata.category(chr(i)).startswith('P')
  }.union(string.punctuation)
  targets = [[normalize_mlqa(t, lang, punct) for t in u] for u in targets]
  predictions = [normalize_mlqa(p, lang, punct) for p in predictions]
  return targets, predictions

def mt5_mlqa_em(targets, predictions, lang=None):
  targets, predictions = _mlqa_metrics_prepare(targets, predictions, lang)
  return qa_em(targets, predictions)

def mt5_mlqa_f1(targets, predictions, lang=None):
  targets, predictions = _mlqa_metrics_prepare(targets, predictions, lang)
  return qa_f1(targets, predictions)
