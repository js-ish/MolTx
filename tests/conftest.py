import os.path
import pytest
from moltx import tokenizers as tkz


@pytest.fixture
def datadir():
    return os.path.join(os.path.dirname(__file__), '../moltx/data')


@pytest.fixture
def tokenizer():
    return tkz.MoltxTokenizer(token_size=16)
