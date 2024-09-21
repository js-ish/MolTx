import typing
import torch
from moltx import tokenizers, models


class Base:
    def __init__(self, tokenizer: tokenizers.MoltxTokenizer, device: torch.device = torch.device('cpu')) -> None:
        self.tokenizer = tokenizer
        self.device = device

    def _tokenize(self, smiles: typing.Sequence[str], seq_len: int = None, spe_dropout: float = 0) -> torch.Tensor:
        tks_list = [self.tokenizer(smi, spe_dropout) for smi in smiles]
        size = seq_len or max(map(len, tks_list))
        out = [self._tokens2tensor(tks, size).unsqueeze(0) for tks in tks_list]
        return torch.concat(out)

    def _tokens2tensor(self, tokens: typing.Sequence[int], size: int) -> torch.Tensor:
        out = torch.zeros(size, dtype=torch.int)
        if len(tokens) > size:
            raise IndexError('the length of tokens is greater than size!')
        for i, tk in enumerate(tokens):
            out[i] = tk
        return out.to(self.device)


class AdaMR(Base):
    def __init__(self, device: torch.device = torch.device('cpu')) -> None:
        tokenizer = tokenizers.MoltxTokenizer.from_pretrain(models.AdaMRTokenizerConfig.Spe)
        super().__init__(tokenizer=tokenizer, device=device)

    def __call__(self, s1: typing.Sequence[str], s2: typing.Sequence[str]) -> typing.Tuple[torch.Tensor]:
        if len(s1) != len(s2):
            raise RuntimeError("the length of s1 and s2 must be the same!")
        src = self._tokenize(s1, spe_dropout=0.2)
        ts2 = self._tokenize(s2, spe_dropout=1.0)
        bos = self._tokenize([self.tokenizer.BOS for _ in range(len(s2))])
        eos = self._tokenize([self.tokenizer.EOS for _ in range(len(s2))])
        tgt = torch.concat([bos, ts2], dim=1)
        out = torch.concat([ts2, eos], dim=1)
        return src, tgt, out


class AdaMRClassifier(AdaMR):
    def __call__(self, smiles: typing.Sequence[str], labels: typing.Sequence[int], seq_len: int = None) -> typing.Tuple[torch.Tensor]:
        if len(smiles) != len(labels):
            raise RuntimeError(
                "the length of smiles and labels must be the same!")
        head = [self.tokenizer.CLS for _ in range(len(smiles))]
        src = self._tokenize(smiles, seq_len)
        tgt = self._tokenize(head)
        out = torch.tensor(labels, device=self.device)
        return src, tgt, out


class AdaMRRegression(AdaMR):
    def __call__(self, smiles: typing.Sequence[str], values: typing.Sequence[float], seq_len: int = None) -> typing.Tuple[torch.Tensor]:
        if len(smiles) != len(values):
            raise RuntimeError(
                "the length of smiles and values must be the same!")
        head = [self.tokenizer.CLS for _ in range(len(smiles))]
        src = self._tokenize(smiles, seq_len)
        tgt = self._tokenize(head)
        out = torch.tensor(values, device=self.device).unsqueeze(-1)
        return src, tgt, out


class AdaMRDistGeneration(AdaMR):
    def __init__(self, device: torch.device = torch.device('cpu')) -> None:
        tokenizer = tokenizers.MoltxTokenizer.from_pretrain(models.AdaMRTokenizerConfig.Atom)
        super(AdaMR, self).__init__(tokenizer=tokenizer, device=device)

    def __call__(self, smiles: typing.Sequence[str], seq_len: int = None) -> typing.Tuple[torch.Tensor]:
        seq_len = seq_len and seq_len - 1
        src = self._tokenize([self.tokenizer.CLS for _ in range(len(smiles))])
        smi = self._tokenize(smiles, seq_len=seq_len)
        bos = self._tokenize([self.tokenizer.BOS for _ in range(len(smiles))])
        eos = self._tokenize([self.tokenizer.EOS for _ in range(len(smiles))])
        tgt = torch.concat([bos, smi], dim=1)
        out = torch.concat([smi, eos], dim=1)
        return src, tgt, out


class AdaMRGoalGeneration(AdaMR):
    def __init__(self, device: torch.device = torch.device('cpu')) -> None:
        tokenizer = tokenizers.MoltxTokenizer.from_pretrain(models.AdaMRTokenizerConfig.Atom)
        super(AdaMR, self).__init__(tokenizer=tokenizer, device=device)

    def __call__(self, smiles: typing.Sequence[str], goals: typing.Sequence[float], seq_len: int = None) -> typing.Tuple[torch.Tensor]:
        if len(smiles) != len(goals):
            raise RuntimeError(
                "the length of smiles and goals must be the same!")
        seq_len = seq_len and seq_len - 1
        src = self._tokenize([self.tokenizer.CLS for _ in range(len(smiles))])
        smi = self._tokenize(smiles, seq_len=seq_len)
        bos = self._tokenize([self.tokenizer.BOS for _ in range(len(smiles))])
        eos = self._tokenize([self.tokenizer.EOS for _ in range(len(smiles))])
        tgt = torch.concat([bos, smi], dim=1)
        out = torch.concat([smi, eos], dim=1)
        goal = torch.tensor(goals, device=self.device).unsqueeze(-1)
        return goal, src, tgt, out


class AdaMR2(Base):
    def __init__(self, device: torch.device = torch.device('cpu')) -> None:
        tokenizer = tokenizers.MoltxTokenizer.from_pretrain(models.AdaMRTokenizerConfig.Spe)
        super().__init__(tokenizer=tokenizer, device=device)

    def __call__(self, s1: typing.Sequence[str], s2: typing.Sequence[str]) -> typing.Tuple[torch.Tensor]:
        if len(s1) != len(s2):
            raise RuntimeError("the length of s1 and s2 must be the same!")
        ts1 = self._tokenize(s1, spe_dropout=0.2)
        ts2 = self._tokenize(s2, spe_dropout=1.0)
        zero = torch.zeros_like(ts1)
        bos = self._tokenize([self.tokenizer.BOS for _ in range(len(s2))])
        eos = self._tokenize([self.tokenizer.EOS for _ in range(len(s2))])
        tgt = torch.concat([ts1, bos, ts2], dim=1)
        out = torch.concat([zero, ts2, eos], dim=1)
        return tgt, out


class AdaMR2Classifier(AdaMR2):
    def __call__(self, smiles: typing.Sequence[str], labels: typing.Sequence[int], seq_len: int = None) -> typing.Tuple[torch.Tensor]:
        if len(smiles) != len(labels):
            raise RuntimeError(
                "the length of smiles and labels must be the same!")
        tgt = self._tokenize(
            [f"{smi}{self.tokenizer.CLS}" for smi in smiles], seq_len)
        out = torch.tensor(labels, device=self.device)
        return tgt, out


class AdaMR2Regression(AdaMR2):
    def __call__(self, smiles: typing.Sequence[str], values: typing.Sequence[float], seq_len: int = None) -> typing.Tuple[torch.Tensor]:
        if len(smiles) != len(values):
            raise RuntimeError(
                "the length of smiles and values must be the same!")
        tgt = self._tokenize(
            [f"{smi}{self.tokenizer.CLS}" for smi in smiles], seq_len)
        out = torch.tensor(values, device=self.device).unsqueeze(-1)
        return tgt, out


class AdaMR2DistGeneration(AdaMR2):
    def __init__(self, device: torch.device = torch.device('cpu')) -> None:
        tokenizer = tokenizers.MoltxTokenizer.from_pretrain(models.AdaMRTokenizerConfig.Atom)
        super(AdaMR2, self).__init__(tokenizer=tokenizer, device=device)

    def __call__(self, smiles: typing.Sequence[str], seq_len: int = None) -> typing.Tuple[torch.Tensor]:
        seq_len = seq_len and seq_len - 2
        head = self._tokenize([self.tokenizer.CLS for _ in range(len(smiles))])
        zero = torch.zeros_like(head)
        smi = self._tokenize(smiles, seq_len=seq_len)
        bos = self._tokenize([self.tokenizer.BOS for _ in range(len(smiles))])
        eos = self._tokenize([self.tokenizer.EOS for _ in range(len(smiles))])
        tgt = torch.concat([head, bos, smi], dim=1)
        out = torch.concat([zero, smi, eos], dim=1)
        return tgt, out


class AdaMR2GoalGeneration(AdaMR2):
    def __init__(self, device: torch.device = torch.device('cpu')) -> None:
        tokenizer = tokenizers.MoltxTokenizer.from_pretrain(models.AdaMRTokenizerConfig.Atom)
        super(AdaMR2, self).__init__(tokenizer=tokenizer, device=device)

    def __call__(self, smiles: typing.Sequence[str], goals: typing.Sequence[float], seq_len: int = None) -> typing.Tuple[torch.Tensor]:
        if len(smiles) != len(goals):
            raise RuntimeError(
                "the length of smiles and goals must be the same!")
        seq_len = seq_len and seq_len - 2
        head = self._tokenize([self.tokenizer.CLS for _ in range(len(smiles))])
        zero = torch.zeros_like(head)
        smi = self._tokenize(smiles, seq_len=seq_len)
        bos = self._tokenize([self.tokenizer.BOS for _ in range(len(smiles))])
        eos = self._tokenize([self.tokenizer.EOS for _ in range(len(smiles))])
        tgt = torch.concat([head, bos, smi], dim=1)
        out = torch.concat([zero, smi, eos], dim=1)
        goal = torch.tensor(goals, device=self.device).unsqueeze(-1)
        return goal, tgt, out
