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
from moltx import datasets, tokenizers
tk = tokenizers.MoltxTokenizer.from_jsonfile(spe_codes=True, token_size=512)
ds = datasets.AdaMR(tokenizer=tk, device=torch.device('cpu'))
generic_smiles = ["C=CC=CC=C", "...."]
canonical_smiles = ["c1cccc1c", "..."]
src, tgt, out = ds(generic_smiles, canonical_smiles)

# train
import torch.nn as nn
from torch.optim import Adam
from moltx import models
conf = models.AdaMRConfig(token_size=512, max_len=256)
model = models.AdaMR(conf)
crt = nn.CrossEntropyLoss(ignore_index=0)
optim = Adam(model.parameters(), lr=0.1)

optim.zero_grad()
pred = model(src, tgt)
loss = crt(pred.view(-1, pred.size(-1)), out.view(-1))
loss.backward()
optim.step()

# save ckpt
torch.save(model.state_dict(), '/path/to/adamr.ckpt')
```


### Finetune


```python
# Classifier finetune
from moltx import datasets, tokenizers
tk = tokenizers.MoltxTokenizer.from_jsonfile(spe_codes=True, token_size=512)

ds = datasets.AdaMRClassifier(tokenizer=tk, device=torch.device('cpu'))
smiles = ["c1cccc1c", "CC[N+](C)(C)Cc1ccccc1Br"]
labels = [0, 1]
src, tgt, out = ds(smiles, labels)

from moltx import models
pretrained_conf = models.AdaMRConfig(token_size=512, max_len=256, ...)
model = models.AdaMRClassifier(num_classes=2, conf=pretrained_conf)
model.load_ckpt('/path/to/adamr.ckpt')
crt = nn.CrossEntropyLoss()
optim = Adam(model.parameters(), lr=0.1)

optim.zero_grad()
pred = model(src, tgt)
loss = crt(pred, out)
loss.backward()
optim.step()

torch.save(model.state_dict(), '/path/to/classifier.ckpt')

# Regression finetune
ds = datasets.AdaMRRegression(tokenizer=tk, device=torch.device('cpu'))
smiles = ["c1cccc1c", "CC[N+](C)(C)Cc1ccccc1Br"]
values = [0.23, 0.12]
src, tgt, out = ds(smiles, values)

model = models.AdaMRRegression(conf=pretrained_conf)
model.load_ckpt('/path/to/adamr.ckpt')
crt = nn.MSELoss()

optim.zero_grad()
pred = model(src, tgt)
loss = crt(pred, out)
loss.backward()
optim.step()

torch.save(model.state_dict(), '/path/to/regression.ckpt')

# Distributed Generation
ds = datasets.AdaMRDistGeneration(tokenizer=tk, device=torch.device('cpu'))
smiles = ["c1cccc1c", "CC[N+](C)(C)Cc1ccccc1Br"]
src, tgt, out = ds(smiles)

model = models.AdaMRDistGeneration(conf=pretrained_conf)
model.load_ckpt('/path/to/adamr.ckpt')
crt = nn.CrossEntropyLoss(ignore_index=0)

optim.zero_grad()
pred = model(src, tgt)
loss = crt(pred.view(-1, pred.size(-1)), out.view(-1))
loss.backward()
optim.step()

torch.save(model.state_dict(), '/path/to/distgen.ckpt')

# Goal Generation
ds = datasets.AdaMRGoalGeneration(tokenizer=tk, device=torch.device('cpu'))
smiles = ["c1cccc1c", "CC[N+](C)(C)Cc1ccccc1Br"]
goals = [0.23, 0.12]
src, tgt, out = ds(smiles, goals)

model = models.AdaMRGoalGeneration(conf=pretrained_conf)
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
from moltx import models, pipelines, tokenizers
tk = tokenizers.MoltxTokenizer.from_jsonfile(spe_codes=True, token_size=512)

# AdaMR
conf = models.AdaMRConfig(...) # use config matching ckpt
model = models.AdaMR(conf)
model.load_ckpt('/path/to/adamr.ckpt')
pipeline = pipelines.AdaMR(tk, model)
pipeline("C=CC=CC=C")
# {"smiles": ["c1ccccc1"], probabilities: [0.9]}

# Classifier
conf = models.AdaMRConfig(...) # use config matching ckpt
model = models.AdaMRClassifier(2, conf)
model.load_ckpt('/path/to/classifier.ckpt')
pipeline = pipelines.AdaMRClassifier(tk, model)
pipeline("C=CC=CC=C")
# {"label": [1], "probability": [0.67]}

# Regression
conf = models.AdaMRConfig(...) # use config matching ckpt
model = models.AdaMRRegression(2, conf)
model.load_ckpt('/path/to/regression.ckpt')
pipeline = pipelines.AdaMRRegression(tk, model)
pipeline("C=CC=CC=C")
# {"value": [0.467], "probability": [0.67]}

# DistGeneration
conf = models.AdaMRConfig(...) # use config matching ckpt
model = models.AdaMRDistGeneration(conf)
model.load_ckpt('/path/to/distgen.ckpt')
pipeline = pipelines.AdaMRDistGeneration(tk, model)
pipeline(k=2)
# {"smiles": ["c1ccccc1", "...."], probabilities: [0.9, 0.1]}

# GoalGeneration
conf = models.AdaMRConfig(...) # use config matching ckpt
model = models.AdaMRGoalGeneration(conf)
model.load_ckpt('/path/to/goalgen.ckpt')
pipeline = pipelines.AdaMRGoalGeneration(tk, model)
pipeline(0.48, k=2)
# {"smiles": ["c1ccccc1", "...."], probabilities: [0.9, 0.1]}
```
