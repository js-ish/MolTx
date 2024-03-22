import torch
import torch.nn as nn
import typing


class AbsPosEncoderDecoder(nn.Module):
    def __init__(self, token_size: int, max_len: int = 512, d_model: int = 768, nhead: int = 8, num_encoder_layers: int = 6, num_decoder_layers: int = 6, dropout: float = 0.1, **kwargs: typing.Any) -> None:
        super().__init__()
        self.token_embedding = nn.Embedding(token_size, d_model, padding_idx=0)
        self.enc_pos_embedding = nn.Embedding(max_len, d_model)
        self.dec_pos_embedding = nn.Embedding(max_len, d_model)
        self.embedding_dropout = nn.Dropout(dropout)
        self.transformer = nn.Transformer(
            d_model, nhead, num_encoder_layers, num_decoder_layers, dropout=dropout, activation='gelu', batch_first=True, **kwargs)
        self.token_output = nn.Linear(d_model, token_size, bias=False)

    def load_ckpt(self, ckpt_files) -> None:
        self.load_state_dict(torch.load(ckpt_files[0], map_location=torch.device('cpu')))

    def _forward_embedding(self, src: torch.Tensor, tgt: torch.Tensor) -> typing.Tuple[torch.Tensor, torch.Tensor]:
        srclen = src.size(-1)
        src = self.token_embedding(src)
        position = torch.arange(0, srclen).to(src.device)
        src += self.enc_pos_embedding(position.unsqueeze(0))

        tgtlen = tgt.size(-1)
        tgt = self.token_embedding(tgt)
        position = torch.arange(0, tgtlen).to(tgt.device)
        tgt += self.dec_pos_embedding(position.unsqueeze(0))

        return self.embedding_dropout(src), self.embedding_dropout(tgt)

    def forward(self, src: torch.Tensor, tgt: torch.Tensor) -> torch.Tensor:
        src, tgt = self._forward_embedding(src, tgt)
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
        self.token_embedding = nn.Embedding(token_size, d_model, padding_idx=0)
        self.pos_embedding = nn.Embedding(max_len, d_model)
        self.embedding_dropout = nn.Dropout(dropout)
        layer = nn.TransformerEncoderLayer(
            d_model, nhead, dropout=dropout, batch_first=True, activation='gelu', norm_first=True)
        self.transformer = nn.TransformerEncoder(layer, num_layers, norm=nn.LayerNorm(d_model))
        self.token_output = nn.Linear(d_model, token_size, bias=False)

    def load_ckpt(self, ckpt_files) -> None:
        self.load_state_dict(torch.load(ckpt_files[0], map_location=torch.device('cpu')))

    def _forward_embedding(self, x: torch.Tensor) -> torch.Tensor:
        xlen = x.size(-1)
        x = self.token_embedding(x)
        position = torch.arange(0, xlen).to(x.device)
        x += self.pos_embedding(position.unsqueeze(0))
        return self.embedding_dropout(x)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self._forward_embedding(x)
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
