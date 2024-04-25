import torch
import pytest
from moltx import nets


def test_abspos_embedding():
    emb = nets.AbsPosEmbedding(token_size=64, max_len=10, d_model=32)
    x = torch.randint(0, 64, (2, 5))
    x = emb(x)
    assert x.shape == (2, 5, 32)
    x = torch.randint(0, 64, (5,))
    x = emb(x)
    assert x.shape == (5, 32)
    x = torch.randint(0, 64, (2, 11))
    with pytest.raises(IndexError):
        emb(x)
    x = torch.randint(64, 96, (2, 5))
    with pytest.raises(IndexError):
        emb(x)
    x = torch.tensor([[1, 2, 0, 0, 0], [1, 2, 3, 4, 0]])
    x = emb(x)
    assert x[0, 2:, :].eq(0).all()
    assert x[1, 4:, :].eq(0).all()


def test_AbsPosEncoderDecoder():
    conf = nets.AbsPosEncoderDecoderConfig(
        token_size=16, max_len=32, d_model=8, nhead=2,
        num_encoder_layers=2, num_decoder_layers=2)
    model = nets.AbsPosEncoderDecoder(conf=conf)
    model.eval()
    model.requires_grad_(False)

    batch = torch.randint(1, 16, (2, 32))
    out = model.forward_(batch, batch)
    assert out.shape == (2, 32, 8)
    feat = model.forward_feature(batch, batch)
    assert feat.shape == (2, 8)
    assert feat.eq(out[:, -1]).all()
    gen = model.forward_generation(batch, batch)
    assert gen.shape == (2, 32, 16)

    tokens = torch.randint(1, 16, (32,))
    out = model.forward_(tokens, tokens)
    assert out.shape == (32, 8)
    feat = model.forward_feature(tokens, tokens)
    assert feat.shape == (8,)
    assert feat.eq(out[-1]).all()
    gen = model.forward_generation(tokens, tokens)
    assert gen.shape == (32, 16)

    conf = nets.AbsPosEncoderDecoderConfig(
        token_size=16, max_len=32, d_model=8, nhead=2,
        num_encoder_layers=2, num_decoder_layers=2, dtype=torch.bfloat16)
    model = nets.AbsPosEncoderDecoder(conf=conf)
    model.eval()
    model.requires_grad_(False)
    
    batch = torch.randint(1, 16, (2, 32))
    out = model.forward_(batch, batch)
    assert out.shape == (2, 32, 8)


def test_AbsPosEncoderCausal():
    conf = nets.AbsPosEncoderCausalConfig(
        token_size=16, max_len=32, d_model=8, nhead=2, num_layers=2)
    model = nets.AbsPosEncoderCausal(conf=conf)
    model.eval()
    model.requires_grad_(False)

    batch = torch.randint(1, 16, (2, 32))
    out = model.forward_(batch)
    assert out.shape == (2, 32, 8)
    feat = model.forward_feature(batch)
    assert feat.shape == (2, 8)
    assert feat.eq(out[:, -1]).all()
    gen = model.forward_generation(batch)
    assert gen.shape == (2, 32, 16)

    tokens = torch.randint(1, 16, (32,))
    out = model.forward_(tokens)
    assert out.shape == (32, 8)
    feat = model.forward_feature(tokens)
    assert feat.shape == (8,)
    assert feat.eq(out[-1]).all()
    gen = model.forward_generation(tokens)
    assert gen.shape == (32, 16)

    conf = nets.AbsPosEncoderCausalConfig(
        token_size=16, max_len=32, d_model=8, nhead=2, num_layers=2, dtype=torch.bfloat16)
    model = nets.AbsPosEncoderCausal(conf=conf)
    model.eval()
    model.requires_grad_(False)

    batch = torch.randint(1, 16, (2, 32))
    out = model.forward_(batch)
    assert out.shape == (2, 32, 8)
