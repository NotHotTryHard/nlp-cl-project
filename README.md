# Code for project "Continual Learning Techniques for Fine-Tuning Language Models"

In continual learning, directly fine-tuning language models on new tasks can lead to catastrophic forgetting. This work investigates methods to fine-tune the T5-small model across a sequence of summarization tasks (CNNDM, WikiHow, XSum) while minimizing forgetting and overfitting. We evaluate several strategies: baseline sequential fine-tuning, data mixing for task retention, and parameter restriction. Additionally, we compare our results to the LFPT5 framework, which uses prompt tuning and regularization-based methods.

by Sergey Sedov and Peter Skovorodnikov
