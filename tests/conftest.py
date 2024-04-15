import os.path
import pytest
from moltx import tokenizers as tkz
from moltx import models


@pytest.fixture
def datadir():
    return os.path.join(os.path.dirname(__file__), '../moltx/data')


@pytest.fixture
def tokenizer():
    return tkz.MoltxTokenizer(token_size=16)


@pytest.fixture
def model_conf():
    return models.AdaMRConfig(
        token_size=16, max_len=32, d_model=8, nhead=2,
        num_encoder_layers=2, num_decoder_layers=2)
