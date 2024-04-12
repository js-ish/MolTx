import typing
import torch
import torch.nn as nn
from moltx import tokenizers


TypeSmiles = typing.Union[str, list[str]]


class Base:
    def __init__(self, device: torch.device, tokenizer: tokenizers.MoltxTokenizer, model: typing.Optional[nn.Module] = None) -> None:
        self.device = device
        self.tokenizer = tokenizer
        if model is not None:
            model = model.to(device)
            model.eval()
            model.requires_grad_(False)
        self.model = model

    def _tokenize(self, smiles: TypeSmiles) -> torch.Tensor:
        if isinstance(smiles, str):
            tks = self.tokenizer(smiles)
            size = len(tks)
            return self._tokens2tensor(tks, size)
        tks_list = [self.tokenizer(smi) for smi in smiles]
        size = max(map(len, tks_list))
        out = [self._tokens2tensor(tks, size).unsqueeze(0) for tks in tks_list]
        return torch.concat(out)

    def _tokens2tensor(self, tokens: list[int], size: int) -> torch.Tensor:
        out = torch.zeros(size, dtype=torch.int, device=self.device)
        if len(tokens) > size:
            raise IndexError('the length of tokens is greater than size!')
        for i, tk in enumerate(tokens):
            out[i] = tk
        return out

    @torch.no_grad()
    def _greedy_search(self, tgt: torch.Tensor, **kwds: torch.Tensor) -> typing.Tuple[list[int], float]:
        maxlen = self.model.conf.max_len
        eos = self.tokenizer[self.tokenizer.EOS]
        log_prob = torch.zeros(1, device=self.device)
        for _ in range(maxlen - tgt.size(-1)):
            next_log_prob, next_token = self.model(tgt=tgt, **kwds)[-1].log_softmax(-1).max(-1, keepdims=True) # [token_size] max-> []
            if next_token.item() == eos:
                break
            log_prob += next_log_prob
            tgt = torch.concat((tgt, next_token), dim=-1)
        return tgt[1:].tolist(), log_prob.exp().item()

    @torch.no_grad()
    def _random_sample(self, tgt: torch.Tensor, temperature=1, **kwds: torch.Tensor):
        maxlen = self.model.conf.max_len
        eos = self.tokenizer[self.tokenizer.EOS]
        log_prob = torch.zeros(1, device=self.device)
        for _ in range(maxlen - tgt.size(-1)):
            next_log_probs = (self.model(tgt=tgt, **kwds)[-1] / temperature).softmax(-1) # [token_size]
            rand_num = torch.rand((), device=self.device)
            next_token = (next_log_probs.cumsum(-1) < rand_num).sum(-1, keepdims=True) # [1]
            if next_token.item() == eos:
                break
            log_prob += next_log_probs[next_token].log()
            tgt = torch.concat((tgt, next_token), dim=-1)
        return tgt[1:].tolist(), log_prob.exp().item()

    @torch.no_grad()
    def _beam_search(self, tgt: torch.Tensor, **kwds: torch.Tensor):
        # tgt: [beam, seqlen]
        # when beam_width == 1, beam search is equal to greedy search
        if tgt.dim() != 2:
            raise RuntimeError("tgt must be batched!")
        maxlen = self.model.conf.max_len
        eos = self.tokenizer[self.tokenizer.EOS]
        token_size = self.model.conf.token_size
        beam_width = tgt.size(0)
        log_probs = torch.zeros(beam_width, 1, device=self.device)
        meet_end = torch.zeros_like(log_probs, dtype=torch.bool)
        for _ in range(maxlen - tgt.size(-1)):
            next_log_probs = self.model(tgt=tgt, **kwds)[-1].log_softmax(-1) # [beam, token_size]
            next_log_probs = (next_log_probs + log_probs).view(-1, 1) # [beam * tokensize, 1]
            log_probs, idx = next_log_probs.topk(k=beam_width, dim=0) # [beam, 1]
            tgt_idx = idx.div(token_size, rounding_mode="floor") # [beam, 1]
            next_tokens = idx - tgt_idx * token_size  # [beam, 1]
            meet_end |= next_tokens.eq(self.eos)
            tgt = tgt[tgt_idx.squeeze()]
            tgt = torch.concat((tgt, next_tokens), dim=-1)
            if meet_end.all():
                break
        probs = log_probs.squeeze().exp().tolist()
        tokens = []
        for line in tgt:
            idx = (line == eos).nonzero(as_tuple=True)[0]
            if idx.numel() > 0:
                tokens.append(line[:idx[0]].tolist())
            else:
                tokens.append(line.tolist())
        return tokens, probs


class AdaMR(Base):

    def prepare_tensors(self, s1: TypeSmiles, s2: typing.Optional[TypeSmiles] = None) -> typing.Any:
        if not isinstance(s1, [str, list]):
            raise TypeError("the type of smiles must be str or str list!")

        src = self._tokenize(s1)

        if s2 is None:
            if isinstance(s1, str):
                tgt = self._tokenize(self.tokenizer.BOS)
            else:
                tgt = self._tokenize([self.tokenizer.BOS for _ in range(len(s1))])
            return (src, tgt), None

        if type(s2) != type(s1):
            raise TypeError("the type of smiles and alias_smiles must be the same!")

        if isinstance(s1, str):
            tgt = self._tokenize(f"{self.tokenizer.BOS}{s2}")
            out = self._tokenize(f"{s2}{self.tokenizer.EOS}")
        else:
            if len(s1) != len(s2):
                raise RuntimeError("the length of smiles and alias_smiles must be the same!")
            tgt = self._tokenize([f"{self.tokenizer.BOS}{smi}" for smi in s2])
            out = self._tokenize([f"{smi}{self.tokenizer.EOS}" for smi in s2])
        return (src, tgt), out

    def __call__(self, smiles: str) -> typing.Mapping:
        (src, tgt), _ = self.prepare_tensors(smiles)
        tkidx, prob = self._greedy_search(src=src, tgt=tgt)
        return {
            'smiles': self.tokenizer.decode(tkidx),
            'probability': prob
        }


class AdaMRClassifier(AdaMR):
    def prepare_tensors(self, smiles: TypeSmiles) -> typing.Any:
        args, _ = super().prepare_tensors(smiles, smiles)
        return args

    def __call__(self, smiles: str) -> TypeSmiles:
        args = self.prepare_tensors(smiles)
        out = self.model(*args)
        prob, klass= out.softmax(-1).max(-1)
        return {
            'class': klass.item(),
            'probability': prob.item()
        }


class AdaMRRegression(AdaMR):
    def prepare_tensors(self, smiles: TypeSmiles) -> typing.Any:
        args, _ = super().prepare_tensors(smiles, smiles)
        return args

    def __call__(self, smiles: str) -> TypeSmiles:
        args = self.prepare_tensors(smiles)
        out = self.model(*args)
        return {
            'value': out.item()
        }


class AdaMRDistGeneration(AdaMR):
    def prepare_tensors(self, smiles: typing.Optional[TypeSmiles] = None) -> typing.Any:
        if smiles is None:
            return super().prepare_tensors(self.tokenizer.CLS)
        if isinstance(smiles, str):
            return super().prepare_tensors(self.tokenizer.CLS, smiles)
        head = [self.tokenizer.CLS for _ in range(len(smiles))]
        return super().prepare_tensors(head, smiles)

    def __call__(self) -> typing.Mapping:
        (src, tgt), _ = self.prepare_tensors()
        tkidx, prob = self._random_sample(src=src, tgt=tgt)
        return {
            'smiles': self.tokenizer.decode(tkidx),
            'probability': prob
        }

class AdaMRGoalGeneration(AdaMR):
    def prepare_tensors(self, smiles: typing.Optional[TypeSmiles] = None) -> typing.Any:
        if smiles is None:
            return super().prepare_tensors(self.tokenizer.CLS)
        if isinstance(smiles, str):
            return super().prepare_tensors(self.tokenizer.CLS, smiles)
        head = [self.tokenizer.CLS for _ in range(len(smiles))]
        return super().prepare_tensors(head, smiles)

    def __call__(self, goal: float) -> typing.Mapping:
        (src, tgt), _ = self.prepare_tensors()
        tkidx, prob = self._random_sample(src=src, tgt=tgt, goal=goal)
        return {
            'smiles': self.tokenizer.decode(tkidx),
            'probability': prob
        }
