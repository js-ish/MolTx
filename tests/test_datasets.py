import pytest
from moltx import datasets, tokenizers as tkz


def test_Base(tokenizer):
    ds = datasets.Base(tokenizer)
    smiles = ["<bos>CC[N+](C)(C)Br", "<bos>c1ccccc1"]
    out = ds._tokenize(smiles)
    assert out.shape == (2, 11)
    assert out[0, 0].item() == out[1, 0].item() == ds.tokenizer[ds.tokenizer.BOS]
    assert out[1, -1].item() == 0


def test_AdaMR():
    tokenizer = tkz.MoltxTokenizer.from_jsonfile(spe_codes=True, token_size=128, dropout=0.5, spe_merges=256)
    ds = datasets.AdaMR(tokenizer)
    s1 = ["CC[N+]CCBr", "c1ccccc1"]
    s2 = ["CC[N+]CCBr"]
    with pytest.raises(RuntimeError):
        ds(s1, s2)
    s2 = ["CC[N+](C)(C)Br", ""]
    src, tgt, out = ds(s1, s2)
    assert src.size(0) == 2 and src.size(1) <= 8
    assert tgt.shape == out.shape
    assert tgt[0, 1:].eq(out[0, :-1]).all()

def test_AdaMRClassifier(tokenizer):
    ds = datasets.AdaMRClassifier(tokenizer)
    smiles = ["CC[N+]CCBr", "Cc1ccc1"]
    labels = [1, 2]
    with pytest.raises(RuntimeError):
        ds(smiles, labels[:1])
    src, tgt, out = ds(smiles, labels)
    assert src.shape == (2, 7)
    assert tgt.shape == (2, 9)
    assert out.shape == (2, 1)

def test_AdaMRRegression(tokenizer):
    ds = datasets.AdaMRRegression(tokenizer)
    smiles = ["CC[N+]CCBr", "Cc1ccc1"]
    values = [1.1, 1.2]
    with pytest.raises(RuntimeError):
        ds(smiles, values[:1])
    src, tgt, out = ds(smiles, values)
    assert src.shape == (2, 7)
    assert tgt.shape == (2, 9)
    assert out.shape == (2, 1)


def test_AdaMRDistGeneration(tokenizer):
    ds = datasets.AdaMRDistGeneration(tokenizer)
    smiles = ["CC[N+]CCBr", "c1ccc1"]
    src, tgt, out = ds(smiles)
    assert src.shape == (2, 1)
    assert tgt.shape == (2, 7)
    assert out.shape == (2, 7)


def test_AdaMRGoalGeneration(tokenizer):
    ds = datasets.AdaMRGoalGeneration(tokenizer)
    smiles = ["CC[N+]CCBr", "c1ccc1"]
    goals = [1.1, 1.2]
    goal, src, tgt, out = ds(smiles, goals)
    assert goal.shape == (2, 1)
    assert src.shape == (2, 1)
    assert tgt.shape == (2, 7)
    assert out.shape == (2, 7)
