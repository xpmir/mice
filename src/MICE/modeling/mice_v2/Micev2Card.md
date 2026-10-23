---
language: en
license: apache-2.0
library_name: sentence-transformers
base_model: {{base}}
model_name: {{model_id}}
source: https://github.com/xpmir/cross-encoders
paper: https://arxiv.org/abs/2602.16299
tags:
- cross-encoder
- mice
- modernbert
- information-retrieval
- reranking
datasets:
- msmarco
pipeline_tag: text-classification
---

# {{model_id}}

[![Paper](https://img.shields.io/badge/Paper-Arxiv-red)](https://arxiv.org/abs/2602.16299)
[![GitHub](https://img.shields.io/badge/GitHub-Code-blue)](https://github.com/xpmir/cross-encoders)

This model is a **MICE** (Minimal Interaction Cross-Encoder) model based on `{{base}}` (ModernBERT backbone).

MICE models separate query and document processing in the bottom layers, enabling independent document contextualization and faster re-ranking with cross-attention only in the top interaction layers.

### Contents
- [Model Description](#model-description)
- [Usage](#usage)
- [Evaluations](#evaluations)


## Model Description

- **Architecture:** MICE v2 (Minimal Interaction Cross-Encoder)
- **Base Model:** {{base}}
- **Total Parameters:** {{total_parameters}}
- **Contextualization Layers:** {{n_contextualization_layers}}
- **Interaction Layers:** {{n_interaction_layers}}
- **Training Strategy:** MSE on [`mxbai-rerank-large-v2`](https://huggingface.co/mixedbread-ai/mxbai-rerank-large-v2) (following the [Ettin reranker Family](https://huggingface.co/blog/ettin-reranker) recipe)
- **Language:** English

## Usage

### 1. With `sentence-transformers`

```python
from sentence_transformers import CrossEncoder

model = CrossEncoder("{{model_id}}", trust_remote_code=True)
scores = model.predict([
    ("What is the capital of France?", "Paris is the capital and most populous city of France.")
])
print(scores)
```

### 2. With Hugging Face `transformers`

```python
import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer

tokenizer = AutoTokenizer.from_pretrained("{{model_id}}")
model = AutoModelForSequenceClassification.from_pretrained("{{model_id}}", trust_remote_code=True)
model.eval()

inputs = tokenizer(["What is the capital of France?"], ["Paris is the capital of France."], return_tensors="pt")
with torch.no_grad():
    logits = model(**inputs).logits
print(logits)
```

### 3. With `xpmir`

```python
import torch
from xpmir.neural.sentence_transformers import STCrossEncoder
from xpmir.letor.records import PointwiseItems

scorer = STCrossEncoder.C(model_id="{{model_id}}").instance()
scorer.initialize()
scorer.eval()

queries = ["What is the capital of France?"]
documents = ["Paris is the capital and most populous city of France."]
input_records = PointwiseItems.from_texts(topics=queries, documents=documents)
with torch.no_grad():
    output = scorer(input_records)
print(output)
```
