import pytest
from moltx import datasets


def test_Base(tokenizer):
    ds = datasets.Base(tokenizer)
    smiles = ["<bos>CC[N+](C)(C)Br", "<bos>c1ccccc1"]
    out = ds._tokenize(smiles)
    assert out.shape == (2, 11)
    assert out[0, 0].item() == out[1, 0].item() == ds.tokenizer[ds.tokenizer.BOS]
    assert out[1, -1].item() == 0


def test_AdaMR(tokenizer):
    ds = datasets.AdaMR(tokenizer)
    s1 = ["<bos>CC[N+]CCBr", "<bos>c1ccc1"]
    s2 = ["<bos>CC[N+]CCBr"]
    with pytest.raises(RuntimeError):
        ds(s1, s2)
    s2 = ["<bos>CC[N+](C)(C)Br", "<bos>c1ccccc1"]
    src, tgt, out = ds(s1, s2)
    assert src.shape == (2, 7)
    assert tgt.shape == out.shape == (2, 12)
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