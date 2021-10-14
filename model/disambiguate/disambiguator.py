#! /usr/bin/env python
"""
Copyright (c) Facebook, Inc. and its affiliates.
All rights reserved.
This source code is licensed under the license found in the LICENSE file in the
root directory of this source tree.

Implementation of Disambiguation Model.

Author(s): Satwik Kottur
"""

from __future__ import absolute_import, division, print_function, unicode_literals
import argparse

from transformers import AutoConfig, GPT2ForSequenceClassification,BartForSequenceClassification
import torch
import torch.nn as nn


class Disambiguator(nn.Module):
    def __init__(self, tokenizer, args):
        super(Disambiguator, self).__init__()
        model_config = AutoConfig.from_pretrained(
            pretrained_model_name_or_path=args.model, num_labels=2
        )
        self.lm = GPT2ForSequenceClassification(model_config) if 'gpt2' in args.model else BartForSequenceClassification(model_config)
        # Fix model padding token id.
        self.lm.resize_token_embeddings(len(tokenizer))
        self.lm.config.pad_token_id = self.lm.config.eos_token_id

        self.args = args
        self.lm.to(args.device)

    def forward(self, batch):
        logits = self.lm(**batch["text_in"])["logits"]
        return logits
