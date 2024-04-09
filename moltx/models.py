import typing
import torch
import torch.nn as nn
from moltx import nets


class AdaMR(nets.AbsPosEncoderDecoder):
    pass


class AdaMRClassifier(nets.AbsPosEncoderDecoder):
    def __init__(self, num_classes: int, *args: typing.Any, **kwargs: typing.Any) -> None:
        conf = nets.AbsPosEncoderDecoderConfig(*args, **kwargs)
        super().__init__(conf=conf)
        d_hidden = conf.d_model // 2
        self.fc = nn.Sequential(
            nn.Dropout(conf.dropout),
            nn.Linear(conf.d_model, d_hidden),
            nn.Tanh(),
            nn.Dropout(conf.dropout),
            nn.Linear(d_hidden, num_classes)
        )

    def load_ckpt(self, ckpt_files: typing.List[str]) -> None:
        self.load_state_dict(torch.load(
            ckpt_files[0], map_location=torch.device('cpu')), strict=False)

    def forward(self, src: torch.Tensor, tgt: torch.Tensor) -> torch.Tensor:
        feats = super().forward_feature(src, tgt)
        return self.fc(feats)


class AdaMRRegression(nets.AbsPosEncoderDecoder):
    def __init__(self, *args: typing.Any, **kwargs: typing.Any) -> None:
        conf = nets.AbsPosEncoderDecoderConfig(*args, **kwargs)
        super().__init__(conf=conf)
        d_hidden = conf.d_model // 2
        self.fc = nn.Sequential(
            nn.Dropout(conf.dropout),
            nn.Linear(conf.d_model, d_hidden),
            nn.Tanh(),
            nn.Dropout(conf.dropout),
            nn.Linear(d_hidden, 1)
        )

    def load_ckpt(self, ckpt_files: typing.List[str]) -> None:
        self.load_state_dict(torch.load(
            ckpt_files[0], map_location=torch.device('cpu')), strict=False)

    def forward(self, src: torch.Tensor, tgt: torch.Tensor) -> torch.Tensor:
        feats = super().forward_feature(src, tgt)
        return self.fc(feats)


class AdaMRDistGeneration(nets.AbsPosEncoderDecoder):
    def forward(self, src: torch.Tensor, tgt: torch.Tensor) -> torch.Tensor:
        return super().forward_generation(src, tgt)


class AdaMRGoalGeneration(nets.AbsPosEncoderDecoder):
    def forward(self, goal: float, src: torch.Tensor, tgt: torch.Tensor) -> torch.Tensor:
        is_batched = src.dim() == 2
        src = self.embedding(src)
        tgt = self.embedding(tgt)
        if is_batched:
            src[:, 0] = src[:, 0] * goal
        else:
            src[0] = src[0] * goal
        out = self.transformer(src, tgt, tgt_is_causal=True)
        return self.token_output(out)