import pytest
from moltx import datasets, models, tokenizers as tkz


def test_Base(tokenizer):
    ds = datasets.Base(tokenizer)
    smiles = ["<bos>CC[N+](C)(C)Br", "<bos>c1ccccc1"]
    out = ds._tokenize(smiles)
    assert out.shape == (2, 11)
    assert out[0, 0].item() == out[1, 0].item() == ds.tokenizer[ds.tokenizer.BOS]
    assert out[1, -1].item() == 0


def test_AdaMR():
    tokenizer = tkz.MoltxTokenizer.from_pretrain(models.AdaMRTokenizerConfig.Pretrain)
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

    seq_len = 10
    src, tgt, out = ds(smiles, labels, seq_len)
    assert src.shape == (2, seq_len)
    assert tgt.shape == (2, seq_len)
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

    seq_len = 10
    src, tgt, out = ds(smiles, values, seq_len)
    assert src.shape == (2, seq_len)
    assert tgt.shape == (2, seq_len)
    assert out.shape == (2, 1)


def test_AdaMRDistGeneration(tokenizer):
    ds = datasets.AdaMRDistGeneration(tokenizer)
    smiles = ["CC[N+]CCBr", "c1ccc1"]
    src, tgt, out = ds(smiles)
    assert src.shape == (2, 1)
    assert tgt.shape == (2, 7)
    assert out.shape == (2, 7)

    seq_len = 8
    src, tgt, out = ds(smiles, seq_len)
    assert src.shape == (2, 1)
    assert tgt.shape == (2, seq_len)
    assert out.shape == (2, seq_len)


def test_AdaMRGoalGeneration(tokenizer):
    ds = datasets.AdaMRGoalGeneration(tokenizer)
    smiles = ["CC[N+]CCBr", "c1ccc1"]
    goals = [1.1, 1.2]
    goal, src, tgt, out = ds(smiles, goals)
    assert goal.shape == (2, 1)
    assert src.shape == (2, 1)
    assert tgt.shape == (2, 7)
    assert out.shape == (2, 7)

    seq_len = 9
    goal, src, tgt, out = ds(smiles, goals, seq_len)
    assert goal.shape == (2, 1)
    assert src.shape == (2, 1)
    assert tgt.shape == (2, seq_len)
    assert out.shape == (2, seq_len)

def test_AdaMR2(tokenizer):
    ds = datasets.AdaMR2(tokenizer)
    s1 = ["CC[N+]CCBr", "c1ccccc1"]
    s2 = ["CC[N+]CCBr"]
    with pytest.raises(RuntimeError):
        ds(s1, s2)
    s2 = ["CC[N+](C)(C)Br", ""]
    tgt, out = ds(s1, s2)
    assert tgt.shape == out.shape
    assert tgt.size(0) == 2 and tgt.size(1) == 17
    assert tgt[0, 7:].eq(out[0, 6:-1]).all()
    assert out[0, :6].eq(0).all()

def test_AdaMR2Classifier(tokenizer):
    ds = datasets.AdaMR2Classifier(tokenizer)
    smiles = ["CC[N+]CCBr", "Cc1ccc1"]
    labels = [1, 2]
    with pytest.raises(RuntimeError):
        ds(smiles, labels[:1])
    tgt, out = ds(smiles, labels)
    assert tgt.shape == (2, 9)
    assert out.shape == (2, 1)

    seq_len = 10
    tgt, out = ds(smiles, labels, seq_len)
    assert tgt.shape == (2, seq_len)
    assert out.shape == (2, 1)

def test_AdaMR2Regression(tokenizer):
    ds = datasets.AdaMR2Regression(tokenizer)
    smiles = ["CC[N+]CCBr", "Cc1ccc1"]
    values = [1.1, 1.2]
    with pytest.raises(RuntimeError):
        ds(smiles, values[:1])
    tgt, out = ds(smiles, values)
    assert tgt.shape == (2, 9)
    assert out.shape == (2, 1)

    seq_len = 10
    tgt, out = ds(smiles, values, seq_len)
    assert tgt.shape == (2, seq_len)
    assert out.shape == (2, 1)


def test_AdaMR2DistGeneration(tokenizer):
    ds = datasets.AdaMR2DistGeneration(tokenizer)
    smiles = ["CC[N+]CCBr", "c1ccc1"]
    tgt, out = ds(smiles)
    assert tgt.shape == (2, 8)
    assert out.shape == (2, 8)
    assert out[0, 0].item() == 0
    assert tgt[0, 2:].eq(out[0, 1:-1]).all()

    seq_len = 9
    tgt, out = ds(smiles, seq_len)
    assert tgt.shape == (2, seq_len)
    assert out.shape == (2, seq_len)
    assert out[0, 0].item() == 0


def test_AdaMR2GoalGeneration(tokenizer):
    ds = datasets.AdaMR2GoalGeneration(tokenizer)
    smiles = ["CC[N+]CCBr", "c1ccc1"]
    goals = [1.1, 1.2]
    goal, tgt, out = ds(smiles, goals)
    assert goal.shape == (2, 1)
    assert tgt.shape == (2, 8)
    assert out.shape == (2, 8)
    assert out[0, 0].item() == 0
    assert tgt[0, 2:].eq(out[0, 1:-1]).all()


    seq_len = 9
    goal, tgt, out = ds(smiles, goals, seq_len)
    assert goal.shape == (2, 1)
    assert tgt.shape == (2, seq_len)
    assert out.shape == (2, seq_len)
    assert out[0, 0].item() == 0
