import typing
import torch
import torch.nn as nn
from moltx import tokenizers


class Base:
    def __init__(self, tokenizer: tokenizers.MoltxTokenizer, model: nn.Module, device: torch.device = torch.device('cpu')) -> None:
        self.tokenizer = tokenizer
        model = model.to(device)
        model.eval()
        model.requires_grad_(False)
        self.model = model
        self.device = device

    def _tokenize(self, smiles: str) -> torch.Tensor:
        tks = self.tokenizer(smiles)
        size = len(tks)
        return self._tokens2tensor(tks, size)

    def _tokens2tensor(self, tokens: typing.Sequence[int], size: int) -> torch.Tensor:
        out = torch.zeros(size, dtype=torch.int)
        if len(tokens) > size:
            raise IndexError('the length of tokens is greater than size!')
        for i, tk in enumerate(tokens):
            out[i] = tk
        return out.to(self.device)

    @torch.no_grad()
    def _greedy_search(self, tgt: torch.Tensor, **kwds: torch.Tensor) -> typing.Tuple[typing.Sequence[int], float]:
        maxlen = self.model.conf.max_len
        eos = self.tokenizer[self.tokenizer.EOS]
        log_prob = torch.zeros(1, device=self.device)
        for _ in range(maxlen - tgt.size(-1)):
            next_log_prob, next_token = self.model(
                tgt=tgt, **kwds)[-1].log_softmax(-1).max(-1, keepdims=True)  # [token_size] max-> []
            if next_token.item() == eos:
                break
            log_prob += next_log_prob
            tgt = torch.concat((tgt, next_token), dim=-1)
        return self.tokenizer.decode(tgt[1:].tolist()), log_prob.exp().item()

    @torch.no_grad()
    def _random_sample(self, tgt: torch.Tensor, temperature=1, **kwds: torch.Tensor):
        maxlen = self.model.conf.max_len
        eos = self.tokenizer[self.tokenizer.EOS]
        log_prob = torch.zeros(1, device=self.device)
        for _ in range(maxlen - tgt.size(-1)):
            next_log_probs = (self.model(tgt=tgt, **kwds)
                              [-1] / temperature).softmax(-1)  # [token_size]
            rand_num = torch.rand((), device=self.device)
            next_token = (next_log_probs.cumsum(-1) <
                          rand_num).sum(-1, keepdims=True)  # [1]
            if next_token.item() == eos:
                break
            log_prob += next_log_probs[next_token].log()
            tgt = torch.concat((tgt, next_token), dim=-1)
        return self.tokenizer.decode(tgt[1:].tolist()), log_prob.exp().item()

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
            next_log_probs = self.model(
                tgt=tgt, **kwds)[:, -1].log_softmax(-1)  # [beam, token_size]
            # [beam * tokensize, 1]
            next_log_probs = (next_log_probs + log_probs).view(-1, 1)
            log_probs, idx = next_log_probs.topk(
                k=beam_width, dim=0)  # [beam, 1]
            tgt_idx = idx.div(token_size, rounding_mode="floor")  # [beam, 1]
            next_tokens = idx - tgt_idx * token_size  # [beam, 1]
            meet_end |= next_tokens.eq(eos)
            tgt = tgt[tgt_idx.squeeze()]
            tgt = torch.concat((tgt, next_tokens), dim=-1)
            if meet_end.all():
                break
        probs = log_probs.squeeze().exp().tolist()
        smiles = []
        for line in tgt:
            idx = (line == eos).nonzero(as_tuple=True)[0]
            if idx.numel() > 0:
                smiles.append(self.tokenizer.decode(line[:idx[0]].tolist()))
            else:
                smiles.append(self.tokenizer.decode(line.tolist()))
        return smiles, probs


class AdaMR(Base):

    def _model_args(self, s1: str, s2: typing.Optional[str] = None) -> typing.Tuple[torch.Tensor]:
        src = self._tokenize(s1)
        if s2 is None:
            tgt = self._tokenize(self.tokenizer.BOS)
            return src, tgt
        tgt = self._tokenize(f"{self.tokenizer.BOS}{s2}")
        return src, tgt

    # gentype: greedy, beam
    def __call__(self, smiles: str, k: int = 1, gentype: str = 'greedy') -> typing.Mapping:
        assert k <= 10
        src, tgt = self._model_args(smiles)
        m = getattr(self, f"_call_{gentype}")
        smis, probs = m(src, tgt, k)
        return {
            'smiles': smis,
            'probabilities': probs
        }

    def _call_greedy(self, src: torch.Tensor, tgt: torch.Tensor, k: int) -> typing.Mapping:
        smiles, probs = [], []
        for _ in range(k):
            smi, prob = self._greedy_search(src=src, tgt=tgt)
            smiles.append(smi)
            probs.append(prob)
        return smiles, probs

    def _call_beam(self, src: torch.Tensor, tgt: torch.Tensor, k: int) -> typing.Mapping:
        src = src.unsqueeze(0).repeat(k, 1)
        tgt = tgt.unsqueeze(0).repeat(k, 1)
        return self._beam_search(src=src, tgt=tgt)


class AdaMRClassifier(AdaMR):
    def _model_args(self, smiles: str) -> typing.Tuple[torch.Tensor]:
        return super()._model_args(smiles, f"{smiles}{self.tokenizer.EOS}")

    def __call__(self, smiles: str) -> typing.Mapping:
        args = self._model_args(smiles)
        out = self.model(*args)
        prob, label = out.softmax(-1).max(-1)
        return {
            'label': label.item(),
            'probability': prob.item()
        }


class AdaMRRegression(AdaMR):
    def _model_args(self, smiles: str) -> typing.Tuple[torch.Tensor]:
        return super()._model_args(smiles, f"{smiles}{self.tokenizer.EOS}")

    def __call__(self, smiles: str) -> typing.Mapping:
        args = self._model_args(smiles)
        out = self.model(*args)
        return {
            'value': out.item()
        }


class AdaMRDistGeneration(AdaMR):
    def _model_args(self) -> typing.Tuple[torch.Tensor]:
        return super()._model_args(self.tokenizer.CLS)

    def __call__(self, k: int = 1) -> typing.Mapping:
        assert k <= 10
        src, tgt = self._model_args()
        smis, probs = [], []
        for _ in range(k):
            smi, prob = self._random_sample(src=src, tgt=tgt)
            smis.append(smi)
            probs.append(prob)
        return {
            'smiles': smis,
            'probabilities': probs
        }


class AdaMRGoalGeneration(AdaMR):
    def _model_args(self) -> typing.Tuple[torch.Tensor]:
        return super()._model_args(self.tokenizer.CLS)

    def __call__(self, goal: float, k: int = 1) -> typing.Mapping:
        assert k <= 10
        src, tgt = self._model_args()
        goal = torch.tensor(goal, device=self.device)
        smis, probs = [], []
        for _ in range(k):
            smi, prob = self._random_sample(goal=goal, src=src, tgt=tgt)
            smis.append(smi)
            probs.append(prob)
        return {
            'smiles': smis,
            'probabilities': probs
        }
