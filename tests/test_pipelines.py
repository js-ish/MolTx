import pytest
from moltx import pipelines, models, nets


@pytest.fixture
def adamr_conf():
    return nets.AbsPosEncoderDecoderConfig(
        token_size=16, max_len=32, d_model=8, nhead=2,
        num_encoder_layers=2, num_decoder_layers=2)


@pytest.fixture
def adamr2_conf():
    return nets.AbsPosEncoderCausalConfig(
        token_size=16, max_len=32, d_model=8, nhead=2,num_layers=2)


def test_AdaMR(tokenizer, adamr_conf):
    pipeline = pipelines.AdaMR(tokenizer, models.AdaMR(adamr_conf))
    out = pipeline("CC[N+](C)(C)Br")
    assert 'smiles' in out and 'probability' in out
    assert isinstance(out['smiles'], str)

    out = pipeline("")
    assert 'smiles' in out and 'probability' in out
    assert isinstance(out['smiles'], str)


def test_AdaMRClassifier(tokenizer, adamr_conf):
    pipeline = pipelines.AdaMRClassifier(
        tokenizer, models.AdaMRClassifier(num_classes=2, conf=adamr_conf))
    out = pipeline("CC[N+](C)(C)Br")
    assert 'label' in out and 'probability' in out
    assert isinstance(out['label'], int) and isinstance(out['probability'], float)
    assert out['label'] in range(2)


def test_AdaMRRegression(tokenizer, adamr_conf):
    pipeline = pipelines.AdaMRRegression(
        tokenizer, models.AdaMRRegression(conf=adamr_conf))
    out = pipeline("CC[N+](C)(C)Br")
    assert 'value' in out
    assert isinstance(out['value'], float)


def test_AdaMRDistGeneration(tokenizer, adamr_conf):
    pipeline = pipelines.AdaMRDistGeneration(
        tokenizer, models.AdaMRDistGeneration(adamr_conf))
    out = pipeline(k=1)
    assert 'smiles' in out and 'probabilities' in out
    assert isinstance(out['smiles'], list) and len(out['smiles']) == 1 and isinstance(out['smiles'][0], str)
    assert isinstance(out['probabilities'], list) and len(
        out['probabilities']) == 1


def test_AdaMRGoalGeneration(tokenizer, adamr_conf):
    pipeline = pipelines.AdaMRGoalGeneration(
        tokenizer, models.AdaMRGoalGeneration(adamr_conf))
    out = pipeline(0.48, k=2)
    assert 'smiles' in out and 'probabilities' in out
    assert isinstance(out['smiles'], list) and len(out['smiles']) == 2 and isinstance(out['smiles'][0], str)
    assert isinstance(out['probabilities'], list) and len(
        out['probabilities']) == 2


def test_AdaMR2(tokenizer, adamr2_conf):
    pipeline = pipelines.AdaMR2(tokenizer, models.AdaMR2(adamr2_conf))
    out = pipeline("CC[N+](C)(C)Br")
    assert 'smiles' in out and 'probability' in out
    assert isinstance(out['smiles'], str)

    out = pipeline("")
    assert 'smiles' in out and 'probability' in out
    assert isinstance(out['smiles'], str)


def test_AdaMR2Classifier(tokenizer, adamr2_conf):
    pipeline = pipelines.AdaMR2Classifier(
        tokenizer, models.AdaMR2Classifier(num_classes=2, conf=adamr2_conf))
    out = pipeline("CC[N+](C)(C)Br")
    assert 'label' in out and 'probability' in out
    assert isinstance(out['label'], int) and isinstance(out['probability'], float)
    assert out['label'] in range(2)


def test_AdaMR2Regression(tokenizer, adamr2_conf):
    pipeline = pipelines.AdaMR2Regression(
        tokenizer, models.AdaMR2Regression(conf=adamr2_conf))
    out = pipeline("CC[N+](C)(C)Br")
    assert 'value' in out
    assert isinstance(out['value'], float)


def test_AdaMR2DistGeneration(tokenizer, adamr2_conf):
    pipeline = pipelines.AdaMR2DistGeneration(
        tokenizer, models.AdaMR2DistGeneration(adamr2_conf))
    out = pipeline(k=1)
    assert 'smiles' in out and 'probabilities' in out
    assert isinstance(out['smiles'], list) and len(out['smiles']) == 1 and isinstance(out['smiles'][0], str)
    assert isinstance(out['probabilities'], list) and len(
        out['probabilities']) == 1


def test_AdaMR2GoalGeneration(tokenizer, adamr2_conf):
    pipeline = pipelines.AdaMR2GoalGeneration(
        tokenizer, models.AdaMR2GoalGeneration(adamr2_conf))
    out = pipeline(0.48, k=2)
    assert 'smiles' in out and 'probabilities' in out
    assert isinstance(out['smiles'], list) and len(out['smiles']) == 2 and isinstance(out['smiles'][0], str)
    assert isinstance(out['probabilities'], list) and len(
        out['probabilities']) == 2
