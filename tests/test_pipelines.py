import pytest
from unittest import mock
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


def test__GenBase(tokenizer, adamr2_conf):
    pipeline = pipelines._GenBase(tokenizer, models.AdaMR2(adamr2_conf))
    tgt = pipeline._tokenize(tokenizer.BOS)
    smi1, prob1 = pipeline._greedy_search(tgt)
    assert isinstance(smi1, str) and isinstance(prob1, float)
    smi2, prob2 = pipeline._greedy_search(tgt)
    assert smi2 == smi1 and prob1 == prob2
    tgt2 = pipeline._tokenize(f"{tokenizer.BOS}C=C")
    smi2, prob2 = pipeline._greedy_search(tgt2)
    assert smi2 != smi1 and prob1 != prob2

    smi1, prob1 = pipeline._random_sample(tgt)
    assert isinstance(smi1, str) and isinstance(prob1, float)
    smi2, prob2 = pipeline._random_sample(tgt)
    assert smi2 != smi1 and prob1 != prob2

    out1 = pipeline._beam_search(tgt, beam_width=2)
    assert isinstance(out1, list) and len(out1) == 2
    assert out1[0][1] >= out1[1][1]
    out2 = pipeline._beam_search(tgt, beam_width=2)
    assert out2 == out1


def test_AdaMR(tokenizer, adamr_conf):
    pipeline = pipelines.AdaMR(tokenizer, models.AdaMR(adamr_conf))

    out = pipeline()
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

    out = pipeline()
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


def test_AdaMR2SuperGeneration(tokenizer):
    conf = nets.AbsPosEncoderCausalConfig(
        token_size=16, max_len=64, d_model=8, nhead=2,num_layers=2)
    pipeline = pipelines.AdaMR2SuperGeneration(
        tokenizer, models.AdaMR2(conf))

    out = pipeline('denovo_generation', n_samples_per_trial=2)
    assert isinstance(out, list) and len(out) == 2

    with mock.patch("moltx.pipelines.AdaMR2SuperGeneration._decode_safe") as mock_f:
        mock_f.return_value = ["Cc1cnc5s1.C14CNC1.C35.O34", "c12cc3ccc1O.C=C[C@H]2N.C3C(=O)O"]
        out = pipeline('linker_generation', side_chains=['N#CCC1CN(S(=O)(=O)C[*])C1', '[*]n3cc(c1ncnc2[nH]ccc12)cn3'], n_samples_per_trial=2)
        assert isinstance(out, list) and len(out) == 2

        out = pipeline('scaffold_morphing', side_chains='N#CCC1CN(S(=O)(=O)C[1*])C1.[2*]n3cc(c1ncnc2[nH]ccc12)cn3', n_samples_per_trial=2)
        assert isinstance(out, list) and len(out) == 2

        out = pipeline('motif_extension', motif='N#CCC1CN(S(=O)(=O)C[*])C1', n_samples_per_trial=2)
        assert isinstance(out, list) and len(out) == 2

        out = pipeline('super_structure', core='O=S4(=O)NC(C1CC2C=CC1C2)Nc3ccccc34', n_samples_per_trial=2)
        assert isinstance(out, list) and len(out) == 2

        out = pipeline('scaffold_decoration', scaffold='O=S4(=O)NC(C1CC2C=CC1C2)Nc3cc([*])c([*])cc34', n_samples_per_trial=2)
        assert isinstance(out, list) and len(out) == 2
