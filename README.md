# MolTx

[![CI](https://github.com/js-ish/MolTx/actions/workflows/test.yml/badge.svg)](https://github.com/js-ish/MolTx/actions/workflows/test.yml?query=branch%3Amain)
[![Coverage Status](https://coveralls.io/repos/github/js-ish/MolTx/badge.svg?branch=main)](https://coveralls.io/github/js-ish/MolTx?branch=main)
![PyPI - Python Version](https://img.shields.io/pypi/pyversions/moltx)

## Installation

```
pip install moltx
```

## Usage

### Pretrain

```python
import torch

# prepare dataset
from moltx import datasets, tokenizers, models
ds = datasets.AdaMR2(device=torch.device('cpu'))
generic_smiles = ["C=CC=CC=C", "...."]
canonical_smiles = ["c1cccc1c", "..."]
tgt, out = ds(generic_smiles, canonical_smiles)

# train
import torch.nn as nn
from torch.optim import Adam
from moltx import nets, models

## use custom config
conf = models.AdaMR2.CONFIG_LARGE # or models.AdaMR2.CONFIG_BASE
model = models.AdaMR2(conf)

crt = nn.CrossEntropyLoss(ignore_index=0)
optim = Adam(model.parameters(), lr=0.1)

optim.zero_grad()
pred = model(tgt)
loss = crt(pred.view(-1, pred.size(-1)), out.view(-1))
loss.backward()
optim.step()

# save ckpt
torch.save(model.state_dict(), '/path/to/adamr.ckpt')
```


### Finetune


```python
# Classifier finetune
from moltx import datasets

seq_len = 256 # max token lens of smiles in datasets, if None, use max token lens in smiles
ds = datasets.AdaMR2Classifier(device=torch.device('cpu'))
smiles = ["c1cccc1c", "CC[N+](C)(C)Cc1ccccc1Br"]
labels = [0, 1]
tgt, out = ds(smiles, labels, seq_len)

from moltx import nets, models
pretrained_conf = models.AdaMR.CONFIG_LARGE # or models.AdaMR.CONFIG_BASE
model = models.AdaMR2Classifier(num_classes=2, conf=pretrained_conf)
model.load_ckpt('/path/to/adamr.ckpt')
crt = nn.CrossEntropyLoss()
optim = Adam(model.parameters(), lr=0.1)

optim.zero_grad()
pred = model(tgt)
loss = crt(pred, out)
loss.backward()
optim.step()

torch.save(model.state_dict(), '/path/to/classifier.ckpt')

# Regression finetune
ds = datasets.AdaMR2Regression(device=torch.device('cpu'))
smiles = ["c1cccc1c", "CC[N+](C)(C)Cc1ccccc1Br"]
values = [0.23, 0.12]
tgt, out = ds(smiles, values, seq_len)

model = models.AdaMR2Regression(conf=pretrained_conf)
model.load_ckpt('/path/to/adamr.ckpt')
crt = nn.MSELoss()

optim.zero_grad()
pred = model(tgt)
loss = crt(pred, out)
loss.backward()
optim.step()

torch.save(model.state_dict(), '/path/to/regression.ckpt')

# Distributed Generation
ds = datasets.AdaMR2DistGeneration(device=torch.device('cpu'))
smiles = ["c1cccc1c", "CC[N+](C)(C)Cc1ccccc1Br"]
tgt, out = ds(smiles, seq_len)

model = models.AdaMR2DistGeneration(conf=pretrained_conf)
model.load_ckpt('/path/to/adamr.ckpt')
crt = nn.CrossEntropyLoss(ignore_index=0)

optim.zero_grad()
pred = model(tgt)
loss = crt(pred.view(-1, pred.size(-1)), out.view(-1))
loss.backward()
optim.step()

torch.save(model.state_dict(), '/path/to/distgen.ckpt')

# Goal Generation
ds = datasets.AdaMR2GoalGeneration(device=torch.device('cpu'))
smiles = ["c1cccc1c", "CC[N+](C)(C)Cc1ccccc1Br"]
goals = [0.23, 0.12]
tgt, out = ds(smiles, goals, seq_len)

model = models.AdaMR2GoalGeneration(conf=pretrained_conf)
model.load_ckpt('/path/to/adamr.ckpt')
crt = nn.CrossEntropyLoss(ignore_index=0)

optim.zero_grad()
pred = model(src, tgt)
loss = crt(pred.view(-1, pred.size(-1)), out.view(-1))
loss.backward()
optim.step()

torch.save(model.state_dict(), '/path/to/goalgen.ckpt')
```

### Inference

```python
from moltx import nets, models, pipelines, tokenizers
# AdaMR
conf = models.AdaMR2.CONFIG_LARGE # or models.AdaMR.CONFIG_BASE
model = models.AdaMR2(conf)
model.load_ckpt('/path/to/adamr.ckpt')
pipeline = pipelines.AdaMR2(model)
pipeline("C=CC=CC=C")
# {"smiles": ["c1ccccc1"], probabilities: [0.9]}

# Classifier
conf = models.AdaMR2.CONFIG_LARGE # or models.AdaMR.CONFIG_BASE
model = models.AdaMR2Classifier(2, conf)
model.load_ckpt('/path/to/classifier.ckpt')
pipeline = pipelines.AdaMR2Classifier(model)
pipeline("C=CC=CC=C")
# {"label": [1], "probability": [0.67]}

# Regression
conf = models.AdaMR2.CONFIG_LARGE # or models.AdaMR.CONFIG_BASE
model = models.AdaMR2Regression(2, conf)
model.load_ckpt('/path/to/regression.ckpt')
pipeline = pipelines.AdaMR2Regression(model)
pipeline("C=CC=CC=C")
# {"value": [0.467], "probability": [0.67]}

# DistGeneration
conf = models.AdaMR2.CONFIG_LARGE # or models.AdaMR.CONFIG_BASE
model = models.AdaMR2DistGeneration(conf)
model.load_ckpt('/path/to/distgen.ckpt')
pipeline = pipelines.AdaMR2DistGeneration(model)
pipeline(k=2)
# {"smiles": ["c1ccccc1", "...."], probabilities: [0.9, 0.1]}

# GoalGeneration
conf = models.AdaMR2.CONFIG_LARGE # or models.AdaMR.CONFIG_BASE
model = models.AdaMR2GoalGeneration(conf)
model.load_ckpt('/path/to/goalgen.ckpt')
pipeline = pipelines.AdaMRGoalGeneration(model)
pipeline(0.48, k=2)
# {"smiles": ["c1ccccc1", "...."], probabilities: [0.9, 0.1]}
```
