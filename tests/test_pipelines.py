import pytest
from moltx import pipelines, models


def test_AdaMR(tokenizer, model_conf):
    pipeline = pipelines.AdaMR(tokenizer, models.AdaMR(model_conf))
    out = pipeline("CC[N+](C)(C)Br", k=1)
    assert 'smiles' in out and 'probabilities' in out
    assert isinstance(out['smiles'], list) and len(out['smiles']) == 1 and isinstance(out['smiles'][0], str)
    assert isinstance(out['probabilities'], list) and len(
        out['probabilities']) == 1
    out = pipeline("CC[N+](C)(C)Br", k=2, gentype='beam')
    assert 'smiles' in out and 'probabilities' in out
    assert isinstance(out['smiles'], list) and len(out['smiles']) == 2 and isinstance(out['smiles'][0], str)
    assert isinstance(out['probabilities'], list) and len(
        out['probabilities']) == 2


def test_AdaMRClassifier(tokenizer, model_conf):
    pipeline = pipelines.AdaMRClassifier(
        tokenizer, models.AdaMRClassifier(num_classes=2, conf=model_conf))
    out = pipeline("CC[N+](C)(C)Br")
    assert 'label' in out and 'probability' in out
    assert isinstance(out['label'], int) and isinstance(out['probability'], float)
    assert out['label'] in range(2)


def test_AdaMRRegression(tokenizer, model_conf):
    pipeline = pipelines.AdaMRRegression(
        tokenizer, models.AdaMRRegression(conf=model_conf))
    out = pipeline("CC[N+](C)(C)Br")
    assert 'value' in out
    assert isinstance(out['value'], float)


def test_AdaMRDistGeneration(tokenizer, model_conf):
    pipeline = pipelines.AdaMRDistGeneration(
        tokenizer, models.AdaMRDistGeneration(model_conf))
    out = pipeline(k=1)
    assert 'smiles' in out and 'probabilities' in out
    assert isinstance(out['smiles'], list) and len(out['smiles']) == 1 and isinstance(out['smiles'][0], str)
    assert isinstance(out['probabilities'], list) and len(
        out['probabilities']) == 1


def test_AdaMRGoalGeneration(tokenizer, model_conf):
    pipeline = pipelines.AdaMRGoalGeneration(
        tokenizer, models.AdaMRGoalGeneration(model_conf))
    out = pipeline(0.48, k=2)
    assert 'smiles' in out and 'probabilities' in out
    assert isinstance(out['smiles'], list) and len(out['smiles']) == 2 and isinstance(out['smiles'][0], str)
    assert isinstance(out['probabilities'], list) and len(
        out['probabilities']) == 2
