import torch
import torch.nn as nn
import typing


class AbsPosEmbedding(nn.Module):
    def __init__(self, token_size: int, max_len: int, d_model: int) -> None:
        super().__init__()
        self.token_embedding = nn.Embedding(token_size, d_model, padding_idx=0)
        self.pos_embedding = nn.Embedding(max_len, d_model, padding_idx=0)
        self.dropout = nn.Dropout(0.1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        xlen = x.size(-1)
        mask = (x > 0).long()
        position = torch.arange(1, xlen+1).to(x.device)
        position = position * mask
        x = self.token_embedding(x)
        x += self.pos_embedding(position)
        return self.dropout(x)


class AbsPosEncoderDecoder(nn.Module):
    def __init__(self, token_size: int, max_len: int = 512, d_model: int = 768, nhead: int = 8, num_encoder_layers: int = 6, num_decoder_layers: int = 6, dropout: float = 0.1, **kwargs: typing.Any) -> None:
        super().__init__()
        self.embedding = AbsPosEmbedding(token_size, max_len, d_model)
        self.transformer = nn.Transformer(
            d_model, nhead, num_encoder_layers, num_decoder_layers, dropout=dropout, activation='gelu', batch_first=True, **kwargs)
        self.token_output = nn.Linear(d_model, token_size, bias=False)

    def load_ckpt(self, ckpt_files) -> None:
        self.load_state_dict(torch.load(
            ckpt_files[0], map_location=torch.device('cpu')))

    def forward(self, src: torch.Tensor, tgt: torch.Tensor) -> torch.Tensor:
        src = self.embedding(src)
        tgt = self.embedding(tgt)
        return self.transformer(src, tgt, tgt_is_causal=True)

    def forward_feature(self, src: torch.Tensor, tgt: torch.Tensor) -> torch.Tensor:
        out = self.forward(src, tgt)
        indices = (tgt > 0).sum(dim=-1, keepdim=True) - 1
        indices = indices.unsqueeze(-1).repeat(*
                                               [1 for _ in range(tgt.dim())], out.shape[-1])
        return torch.gather(input=out, dim=-2, index=indices).squeeze()

    def forward_generation(self, src: torch.Tensor, tgt: torch.Tensor) -> torch.Tensor:
        out = self.forward(src, tgt)
        return self.token_output(out)


class AbsPosEncoderCausal(nn.Module):
    def __init__(self, token_size: int, max_len: int = 200, d_model: int = 768, nhead: int = 8, num_layers: int = 12, dropout: float = 0.1, **kwargs: typing.Any) -> None:
        super().__init__()
        self.embedding = AbsPosEmbedding(token_size, max_len, d_model)
        layer = nn.TransformerEncoderLayer(
            d_model, nhead, dropout=dropout, batch_first=True, activation='gelu', norm_first=True)
        self.transformer = nn.TransformerEncoder(
            layer, num_layers, norm=nn.LayerNorm(d_model))
        self.token_output = nn.Linear(d_model, token_size, bias=False)

    def load_ckpt(self, ckpt_files) -> None:
        self.load_state_dict(torch.load(
            ckpt_files[0], map_location=torch.device('cpu')))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.embedding(x)
        return self.transformer(x, is_causal=True)

    def forward_feature(self, x: torch.Tensor) -> torch.Tensor:
        out = self.forward(x)
        indices = (x > 0).sum(dim=-1, keepdim=True) - 1
        indices = indices.unsqueeze(-1).repeat(*
                                               [1 for _ in range(x.dim())], out.shape[-1])
        return torch.gather(input=out, dim=-2, index=indices).squeeze()

    def forward_generation(self, x: torch.Tensor) -> torch.Tensor:
        out = self.forward(x)
        return self.token_output(out)
