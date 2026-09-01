---
language: en
license: apache-2.0
library_name: transformers
base_model: {{base}}
model_name: {{model_id}}
source: https://github.com/xpmir/cross-encoders
paper: http://arxiv.org/abs/2603.03010
tags:
- cross-encoder
- sequence-classification
- tensorboard
datasets:
- msmarco
pipeline_tag: text-classification
---

# {{model_id}}

[![Paper](https://img.shields.io/badge/Paper-Arxiv-red)](http://arxiv.org/abs/2603.03010)
[![All Models](https://img.shields.io/badge/🤗%20Hugging%20Face%20Models-blue)](https://huggingface.co/collections/xpmir/reproducing-cross-encoders)
[![GitHub](https://img.shields.io/badge/GitHub-Code-blue)](https://github.com/xpmir/cross-encoders)

This model is a cross-encoder based on `{{base}}`. It was trained on Ms-Marco using loss `{{loss}}` as part of a reproducibility paper for training cross encoders: "**[Reproducing and Comparing Distillation Techniques for Cross-Encoders](http://arxiv.org/abs/2603.03010)**", see the paper for more details.


### Contents
- [Model Description](#model-description)
- [Usage](#usage)
- [Evals](#evaluations)


## Model Description

This model is intended for **re-ranking** the top results returned by a retrieval system (like BM25, Bi-Encoders or SPLADE).

- **Training Data:** MS MARCO Passage
- **Language:** English
- **Loss** {{loss}}

Training can be easily reproduced using the assiciated repository.
The exact training configuration used for this model is also detailed in [config.yaml](./config.yaml).

## Usage

Quick Start:
```python
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch

tokenizer = AutoTokenizer.from_pretrained("{{base}}")
model = AutoModelForSequenceClassification.from_pretrained("xpmir/{{model_id}}")

features = tokenizer("What is experimaestro ?", "Experimaestro is a powerful framework for ML experiments management...", padding=True, truncation=True, return_tensors="pt")

model.eval()
with torch.no_grad():
    scores = model(**features).logits
    print(scores)
```

## Evaluations

We provide evaluations of this cross-encoder re-ranking the top `{{k}}` documents retrieved by `{{retriever}}`.

{{results}}
