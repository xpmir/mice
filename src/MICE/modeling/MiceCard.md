---
language: en
license: apache-2.0
library_name: transformers
base_model: {{base}}
model_name: {{model_id}}
source: https://github.com/xpmir/cross-encoders
paper: https://arxiv.org/abs/2602.16299
tags:
- cross-encoder
- mice
- information-retrieval
- reranking
datasets:
- msmarco
pipeline_tag: text-classification
---

# {{model_id}}

[![Paper](https://img.shields.io/badge/Paper-Arxiv-red)](https://arxiv.org/abs/2602.16299)
[![GitHub](https://img.shields.io/badge/GitHub-Code-blue)](https://github.com/xpmir/cross-encoders)

This model is a **MICE** (Modular Interaction Cross-Encoder) based on `{{base}}`.

MICE models provide a modular architecture that separates query and document processing, allowing for more efficient inference in some scenarios while maintaining high ranking performance.

### Contents
- [Model Description](#model-description)
- [Usage](#usage)
- [Evals](#evaluations)


## Model Description

- **Architecture:** MICE (Minimal Interaction Cross-Encoder)
- **Base Model:** {{base}}
- **Contextualization Layers:** {{n_contextualization_layers}}
- **Interaction Layers:** {{n_interaction_layers}}
- **Training Data:** MS MARCO Passage
- **Language:** English
- **Loss:** {{loss}}

## Usage

This model uses the `xpmir` library for inference.

```python
from xpm_torch.huggingface import TorchHFHub
import torch

# Load the model
model = TorchHFHub.from_pretrained("xpmir/{{model_id}}")
model.initialize()
model.eval()

# For inference, you can use the model directly with xpmir PointwiseItems
# from xpmir.letor.records import PointwiseItems
# queries = ["What is the capital of France?"]
# documents = ["Paris is the capital and most populous city of France."]
# input_records = PointwiseItems.from_texts(topics=queries, documents=documents)
# with torch.no_grad():
#     output = model(input_records)
```

## Evaluations

We provide evaluations of this cross-encoder re-ranking the top `{{k}}` documents retrieved by `{{retriever}}`.

{{results}}
