import torch
import torch.nn as nn
from moltx import nets, tokenizers


class AdaMRTokenizerConfig:
    Pretrain = tokenizers.MoltxPretrainConfig(
        token_size=512,
        fmt='smiles',
        spe=True,
        spe_dropout=0.2,
        spe_merges=240
    )

    Generation = tokenizers.MoltxPretrainConfig(
        token_size=512,
        fmt='smiles',
        spe=False,
        spe_dropout=1.0,
        spe_merges=240
    )

    Prediction = tokenizers.MoltxPretrainConfig(
        token_size=512,
        fmt='smiles',
        spe=True,
        spe_dropout=0.0,
        spe_merges=240
    )


class AdaMR(nets.AbsPosEncoderDecoder):
    CONFIG_LARGE = nets.AbsPosEncoderDecoderConfig(
        token_size=512,  # max(spe_merge) = 240
        max_len=512,
        d_model=768,
        nhead=12,
        num_encoder_layers=12,
        num_decoder_layers=12,
        dropout=0.1,
        dtype=torch.bfloat16
    )

    CONFIG_BASE = nets.AbsPosEncoderDecoderConfig(
        token_size=512,  # max(spe_merge) = 240
        max_len=512,
        d_model=768,
        nhead=8,
        num_encoder_layers=6,
        num_decoder_layers=6,
        dropout=0.1,
        dtype=torch.float32
    )

    def __init__(self, conf: nets.AbsPosEncoderDecoderConfig = CONFIG_LARGE) -> None:
        super().__init__(conf=conf)

    def forward(self, src: torch.Tensor, tgt: torch.Tensor) -> torch.Tensor:
        return super().forward_generation(src, tgt)


class AdaMRClassifier(AdaMR):
    def __init__(self, num_classes: int, conf: nets.AbsPosEncoderDecoderConfig) -> None:
        super().__init__(conf=conf)
        d_hidden = conf.d_model // 2
        self.fc = nn.Sequential(
            nn.Dropout(conf.dropout),
            nn.Linear(conf.d_model, d_hidden, dtype=conf.dtype),
            nn.Tanh(),
            nn.Dropout(conf.dropout),
            nn.Linear(d_hidden, num_classes, dtype=conf.dtype)
        )

    def load_ckpt(self, *ckpt_files: str) -> None:
        self.load_state_dict(torch.load(
            ckpt_files[0], map_location=torch.device('cpu')), strict=False)

    def forward(self, src: torch.Tensor, tgt: torch.Tensor) -> torch.Tensor:
        feats = super().forward_feature(src, tgt)
        return self.fc(feats)


class AdaMRRegression(AdaMR):
    def __init__(self, conf: nets.AbsPosEncoderDecoderConfig) -> None:
        super().__init__(conf=conf)
        d_hidden = conf.d_model // 2
        self.fc = nn.Sequential(
            nn.Dropout(conf.dropout),
            nn.Linear(conf.d_model, d_hidden, dtype=conf.dtype),
            nn.Tanh(),
            nn.Dropout(conf.dropout),
            nn.Linear(d_hidden, 1, dtype=conf.dtype)
        )

    def load_ckpt(self, *ckpt_files: str) -> None:
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


class AdaMR2(nets.AbsPosEncoderCausal):
    CONFIG_LARGE = nets.AbsPosEncoderCausalConfig(
        token_size=512,  # max(spe_merge) = 240
        max_len=512,
        d_model=768,
        nhead=12,
        num_layers=12,
        dropout=0.1,
        dtype=torch.float32
    )

    CONFIG_BASE = nets.AbsPosEncoderCausalConfig(
        token_size=512,  # max(spe_merge) = 240
        max_len=512,
        d_model=768,
        nhead=8,
        num_layers=6,
        dropout=0.1,
        dtype=torch.float32
    )

    def __init__(self, conf: nets.AbsPosEncoderCausalConfig = CONFIG_LARGE) -> None:
        super().__init__(conf=conf)

    def forward(self, tgt: torch.Tensor) -> torch.Tensor:
        return super().forward_generation(tgt)


class AdaMR2Classifier(AdaMR2):
    def __init__(self, num_classes: int, conf: nets.AbsPosEncoderDecoderConfig) -> None:
        super().__init__(conf=conf)
        d_hidden = conf.d_model // 2
        self.fc = nn.Sequential(
            nn.Dropout(conf.dropout),
            nn.Linear(conf.d_model, d_hidden, dtype=conf.dtype),
            nn.Tanh(),
            nn.Dropout(conf.dropout),
            nn.Linear(d_hidden, num_classes, dtype=conf.dtype)
        )

    def load_ckpt(self, *ckpt_files: str) -> None:
        self.load_state_dict(torch.load(
            ckpt_files[0], map_location=torch.device('cpu')), strict=False)

    def forward(self, tgt: torch.Tensor) -> torch.Tensor:
        feats = super().forward_feature(tgt)
        return self.fc(feats)


class AdaMR2Regression(AdaMR2):
    def __init__(self, conf: nets.AbsPosEncoderDecoderConfig) -> None:
        super().__init__(conf=conf)
        d_hidden = conf.d_model // 2
        self.fc = nn.Sequential(
            nn.Dropout(conf.dropout),
            nn.Linear(conf.d_model, d_hidden, dtype=conf.dtype),
            nn.Tanh(),
            nn.Dropout(conf.dropout),
            nn.Linear(d_hidden, 1, dtype=conf.dtype)
        )

    def load_ckpt(self, *ckpt_files: str) -> None:
        self.load_state_dict(torch.load(
            ckpt_files[0], map_location=torch.device('cpu')), strict=False)

    def forward(self, tgt: torch.Tensor) -> torch.Tensor:
        feats = super().forward_feature(tgt)
        return self.fc(feats)


class AdaMR2DistGeneration(AdaMR2):
    def forward(self, tgt: torch.Tensor) -> torch.Tensor:
        return super().forward_generation(tgt)


class AdaMR2GoalGeneration(AdaMR2):
    def forward(self, goal: torch.Tensor, tgt: torch.Tensor) -> torch.Tensor:
        is_batched = tgt.dim() == 2
        mask = nn.Transformer.generate_square_subsequent_mask(
            tgt.size(-1), device=tgt.device)
        tgt = self.embedding(tgt)
        if is_batched:
            tgt[:, 0] = tgt[:, 0] * goal
        else:
            tgt[0] = tgt[0] * goal
        out = self.transformer(tgt, mask=mask, is_causal=True)
        return self.token_output(out)
