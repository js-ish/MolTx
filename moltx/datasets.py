import typing
import torch
from moltx import tokenizers


class Base:
    def __init__(self, tokenizer: tokenizers.MoltxTokenizer, device: torch.device = torch.device('cpu')) -> None:
        self.tokenizer = tokenizer
        self.device = device

    def _tokenize(self, smiles: typing.Sequence[str]) -> torch.Tensor:
        tks_list = [self.tokenizer(smi) for smi in smiles]
        size = max(map(len, tks_list))
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
    def __call__(self, s1: typing.Sequence[str], s2: typing.Sequence[str]) -> typing.Tuple[torch.Tensor]:
        if len(s1) != len(s2):
            raise RuntimeError("the length of s1 and s2 must be the same!")
        bos = self.tokenizer[self.tokenizer.BOS]
        eos = self.tokenizer[self.tokenizer.EOS]
        src = self._tokenize(s1)
        s2tokens = [self.tokenizer(smi) for smi in s2]
        size = max(map(len, s2tokens)) + 1
        tgt = [self._tokens2tensor([bos] + tks, size).unsqueeze(0) for tks in s2tokens]
        out = [self._tokens2tensor(tks + [eos], size).unsqueeze(0) for tks in s2tokens]
        return src, torch.concat(tgt), torch.concat(out)


class AdaMRClassifier(AdaMR):
    def __call__(self, smiles: typing.Sequence[str], labels: typing.Sequence[int]) -> typing.Tuple[torch.Tensor]:
        if len(smiles) != len(labels):
            raise RuntimeError(
                "the length of smiles and labels must be the same!")
        src = self._tokenize(smiles)
        tgt = self._tokenize(
            [f"{self.tokenizer.BOS}{smi}{self.tokenizer.EOS}" for smi in smiles])
        out = torch.tensor(labels, device=self.device).unsqueeze(-1)
        return src, tgt, out


class AdaMRRegression(AdaMR):
    def __call__(self, smiles: typing.Sequence[str], values: typing.Sequence[float]) -> typing.Tuple[torch.Tensor]:
        if len(smiles) != len(values):
            raise RuntimeError(
                "the length of smiles and values must be the same!")
        src = self._tokenize(smiles)
        tgt = self._tokenize(
            [f"{self.tokenizer.BOS}{smi}{self.tokenizer.EOS}" for smi in smiles])
        out = torch.tensor(values, device=self.device).unsqueeze(-1)
        return src, tgt, out


class AdaMRDistGeneration(AdaMR):
    def __call__(self, smiles: typing.Sequence[str]) -> typing.Tuple[torch.Tensor]:
        head = [self.tokenizer.CLS for _ in range(len(smiles))]
        return super().__call__(head, smiles)


class AdaMRGoalGeneration(AdaMR):
    def __call__(self, smiles: typing.Sequence[str], goals: typing.Sequence[float]) -> typing.Tuple[torch.Tensor]:
        if len(smiles) != len(goals):
            raise RuntimeError(
                "the length of smiles and goals must be the same!")
        head = [self.tokenizer.CLS for _ in range(len(smiles))]
        src, tgt, out = super().__call__(head, smiles)
        goal = torch.tensor(goals, device=self.device).unsqueeze(-1)
        return goal, src, tgt, out


class AdaMR2(Base):
    def __call__(self, s1: typing.Sequence[str], s2: typing.Sequence[str]) -> typing.Tuple[torch.Tensor]:
        if len(s1) != len(s2):
            raise RuntimeError("the length of s1 and s2 must be the same!")
        bos = self.tokenizer[self.tokenizer.BOS]
        eos = self.tokenizer[self.tokenizer.EOS]
        s1tokens = [self.tokenizer(smi) for smi in s1]
        s2tokens = [self.tokenizer(smi) for smi in s2]
        tgt_tokens = [tks1 + [bos] + tks2 for tks1, tks2 in zip(s1tokens, s2tokens)]
        out_tokens = [[0] * len(tks1) + tks2 + [eos] for tks1, tks2 in zip(s1tokens, s2tokens)]
        size = max(map(len, out_tokens))
        tgt = torch.concat([self._tokens2tensor(tks, size).unsqueeze(0) for tks in tgt_tokens])
        out = torch.concat([self._tokens2tensor(tks, size).unsqueeze(0) for tks in out_tokens])
        return tgt, out


class AdaMR2Classifier(AdaMR2):
    def __call__(self, smiles: typing.Sequence[str], labels: typing.Sequence[int]) -> typing.Tuple[torch.Tensor]:
        if len(smiles) != len(labels):
            raise RuntimeError(
                "the length of smiles and labels must be the same!")
        tgt = self._tokenize(
            [f"{self.tokenizer.BOS}{smi}{self.tokenizer.EOS}" for smi in smiles])
        out = torch.tensor(labels, device=self.device).unsqueeze(-1)
        return tgt, out


class AdaMR2Regression(AdaMR2):
    def __call__(self, smiles: typing.Sequence[str], values: typing.Sequence[float]) -> typing.Tuple[torch.Tensor]:
        if len(smiles) != len(values):
            raise RuntimeError(
                "the length of smiles and values must be the same!")
        tgt = self._tokenize(
            [f"{self.tokenizer.BOS}{smi}{self.tokenizer.EOS}" for smi in smiles])
        out = torch.tensor(values, device=self.device).unsqueeze(-1)
        return tgt, out


class AdaMR2DistGeneration(AdaMR2):
    def __call__(self, smiles: typing.Sequence[str]) -> typing.Tuple[torch.Tensor]:
        head = [self.tokenizer.CLS for _ in range(len(smiles))]
        return super().__call__(head, smiles)


class AdaMR2GoalGeneration(AdaMR2):
    def __call__(self, smiles: typing.Sequence[str], goals: typing.Sequence[float]) -> typing.Tuple[torch.Tensor]:
        if len(smiles) != len(goals):
            raise RuntimeError(
                "the length of smiles and goals must be the same!")
        head = [self.tokenizer.CLS for _ in range(len(smiles))]
        tgt, out = super().__call__(head, smiles)
        goal = torch.tensor(goals, device=self.device).unsqueeze(-1)
        return goal, tgt, out
