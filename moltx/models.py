import typing
import torch
import torch.nn as nn
from moltx import nets


AdaMRConfig = nets.AbsPosEncoderDecoderConfig


class AdaMR(nets.AbsPosEncoderDecoder):
    def forward(self, src: torch.Tensor, tgt: torch.Tensor) -> torch.Tensor:
        return super().forward_generation(src, tgt)


class AdaMRClassifier(AdaMR):
    def __init__(self, num_classes: int, conf: AdaMRConfig) -> None:
        super().__init__(conf=conf)
        d_hidden = conf.d_model // 2
        self.fc = nn.Sequential(
            nn.Dropout(conf.dropout),
            nn.Linear(conf.d_model, d_hidden),
            nn.Tanh(),
            nn.Dropout(conf.dropout),
            nn.Linear(d_hidden, num_classes)
        )

    def load_ckpt(self, ckpt_files: typing.Sequence[str]) -> None:
        self.load_state_dict(torch.load(
            ckpt_files[0], map_location=torch.device('cpu')), strict=False)

    def forward(self, src: torch.Tensor, tgt: torch.Tensor) -> torch.Tensor:
        feats = super().forward_feature(src, tgt)
        return self.fc(feats)


class AdaMRRegression(AdaMR):
    def __init__(self, conf: AdaMRConfig) -> None:
        super().__init__(conf=conf)
        d_hidden = conf.d_model // 2
        self.fc = nn.Sequential(
            nn.Dropout(conf.dropout),
            nn.Linear(conf.d_model, d_hidden),
            nn.Tanh(),
            nn.Dropout(conf.dropout),
            nn.Linear(d_hidden, 1)
        )

    def load_ckpt(self, ckpt_files: typing.Sequence[str]) -> None:
        self.load_state_dict(torch.load(
            ckpt_files[0], map_location=torch.device('cpu')), strict=False)

    def forward(self, src: torch.Tensor, tgt: torch.Tensor) -> torch.Tensor:
        feats = super().forward_feature(src, tgt)
        return self.fc(feats)


class AdaMRDistGeneration(AdaMR):
    def forward(self, src: torch.Tensor, tgt: torch.Tensor) -> torch.Tensor:
        return super().forward_generation(src, tgt)


class AdaMRGoalGeneration(AdaMR):
    def forward(self, goal: torch.Tensor, src: torch.Tensor, tgt: torch.Tensor) -> torch.Tensor:
        is_batched = src.dim() == 2
        mask = nn.Transformer.generate_square_subsequent_mask(
            tgt.size(-1), device=tgt.device)
        src = self.embedding(src)
        tgt = self.embedding(tgt)
        if is_batched:
            src[:, 0] = src[:, 0] * goal
        else:
            src[0] = src[0] * goal
        out = self.transformer(src, tgt, tgt_mask=mask, tgt_is_causal=True)
        return self.token_output(out)
