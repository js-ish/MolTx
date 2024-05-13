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
        prefixlen = tgt.size(-1)
        eos = self.tokenizer[self.tokenizer.EOS]
        log_prob = torch.zeros(1, device=self.device)
        for _ in range(maxlen - tgt.size(-1)):
            next_log_prob, next_token = self.model(
                tgt=tgt, **kwds)[-1].log_softmax(-1).max(-1, keepdims=True)  # [token_size] max-> []
            if next_token.item() == eos:
                break
            log_prob += next_log_prob
            tgt = torch.concat((tgt, next_token), dim=-1)
        return self.tokenizer.decode(tgt[prefixlen:].tolist()), log_prob.exp().item()

    @torch.no_grad()
    def _random_sample(self, tgt: torch.Tensor, temperature=1, **kwds: torch.Tensor):
        maxlen = self.model.conf.max_len
        prefixlen = tgt.size(-1)
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
        return self.tokenizer.decode(tgt[prefixlen:].tolist()), log_prob.exp().item()

    @torch.no_grad()
    def _beam_search(self, tgt: torch.Tensor, beam_width: int = 3, **kwds: torch.Tensor):
        # tgt: [seqlen]
        # when beam_width == 1, beam search is equal to greedy search
        maxlen = self.model.conf.max_len
        prefixlen = tgt.size(-1)
        eos = self.tokenizer[self.tokenizer.EOS]
        token_size = self.model.conf.token_size
        smiles = []
        probs = []
        log_probs, next_tokens = self.model(tgt=tgt, **kwds)[-1].log_softmax(-1).topk(k=beam_width, dim=0)  # [beam]
        tgt = torch.concat((tgt.unsqueeze(0).repeat(beam_width, 1), next_tokens.unsqueeze(-1)), dim=-1)
        if 'src' in kwds:
            kwds['src'] = kwds['src'].unsqueeze(0).repeat(beam_width, 1)
        log_probs = log_probs.unsqueeze(-1)
        for _ in range(maxlen - tgt.size(-1)):
            next_log_probs = self.model(
                tgt=tgt, **kwds)[:, -1].log_softmax(-1)  # [beam, token_size]
            next_log_probs = (next_log_probs + log_probs).view(-1, 1)  # [beam * tokensize, 1]
            log_probs, idx = next_log_probs.topk(
                k=beam_width, dim=0)  # [beam, 1]
            tgt_idx = idx.div(token_size, rounding_mode="floor")  # [beam, 1]
            next_tokens = idx - tgt_idx * token_size  # [beam, 1]
            meet_end = (next_tokens.squeeze(1).eq(eos).nonzero()).squeeze(1)
            if meet_end.numel() > 0:
                beam_width -= meet_end.size(0)
                probs.extend(log_probs.index_select(0, meet_end).squeeze(1).exp().tolist())
                end_tgt = tgt.index_select(0, tgt_idx.index_select(0, meet_end).squeeze(1))
                smiles.extend(map(self.tokenizer.decode, end_tgt[:, prefixlen:].tolist()))
                if beam_width == 0:
                    return sorted(zip(smiles, probs), key=lambda x: x[1], reverse=True)
            not_end = (next_tokens.squeeze(1).ne(eos).nonzero()).squeeze(1)
            log_probs = log_probs.index_select(0, not_end)
            next_tokens = next_tokens.index_select(0, not_end)
            tgt = tgt.index_select(0, tgt_idx.index_select(0, not_end).squeeze(1))
            tgt = torch.concat((tgt, next_tokens), dim=-1)
            if 'src' in kwds:
                kwds['src'] = kwds['src'].index_select(0, tgt_idx.index_select(0, not_end).squeeze(1))
        probs.extend(log_probs.squeeze(1).exp().tolist())
        smiles.extend(map(self.tokenizer.decode, tgt[:, prefixlen:].tolist()))
        return sorted(zip(smiles, probs), key=lambda x: x[1], reverse=True)


class AdaMR(Base):

    def _model_args(self, smiles: str) -> typing.Tuple[torch.Tensor]:
        src = self._tokenize(smiles)
        tgt = self._tokenize(self.tokenizer.BOS)
        return src, tgt

    # gentype: greedy, beam
    def __call__(self, smiles: str = "") -> typing.Mapping:
        src, tgt = self._model_args(smiles)
        if len(smiles) > 0:
            meth = self._do_canonicalize
        else:
            meth = self._do_generate
        smi, prob = meth(src, tgt)
        return {
            'smiles': smi,
            'probability': prob
        }

    def _do_generate(self, src: torch.Tensor, tgt: torch.Tensor) -> typing.Mapping:
        return self._random_sample(src=src, tgt=tgt)

    def _do_canonicalize(self, src: torch.Tensor, tgt: torch.Tensor) -> typing.Mapping:
        out = self._beam_search(src=src, tgt=tgt, beam_width=3)
        return out[0]


class AdaMRClassifier(AdaMR):
    def _model_args(self, smiles: str) -> typing.Tuple[torch.Tensor]:
        src = self._tokenize(smiles)
        tgt = self._tokenize(f"{self.tokenizer.BOS}{smiles}{self.tokenizer.EOS}")
        return src, tgt

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
        src = self._tokenize(smiles)
        tgt = self._tokenize(f"{self.tokenizer.BOS}{smiles}{self.tokenizer.EOS}")
        return src, tgt

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


class AdaMR2(Base):

    # gentype: greedy, beam
    def __call__(self, smiles: str = "") -> typing.Mapping:
        tgt = self._tokenize(f"{smiles}{self.tokenizer.BOS}")
        if len(smiles) > 0:
            meth = self._do_canonicalize
        else:
            meth = self._do_generate

        smi, prob = meth(tgt)
        return {
            'smiles': smi,
            'probability': prob
        }

    def _do_generate(self, tgt: torch.Tensor) -> typing.Mapping:
        return self._random_sample(tgt=tgt)

    def _do_canonicalize(self, tgt: torch.Tensor) -> typing.Mapping:
        out = self._beam_search(tgt=tgt)
        return out[0]


class AdaMR2Classifier(AdaMR):

    def __call__(self, smiles: str) -> typing.Mapping:
        tgt = self._tokenize(f"{self.tokenizer.BOS}{smiles}{self.tokenizer.EOS}")
        out = self.model(tgt)
        prob, label = out.softmax(-1).max(-1)
        return {
            'label': label.item(),
            'probability': prob.item()
        }


class AdaMR2Regression(AdaMR):

    def __call__(self, smiles: str) -> typing.Mapping:
        tgt = self._tokenize(f"{self.tokenizer.BOS}{smiles}{self.tokenizer.EOS}")
        out = self.model(tgt)
        return {
            'value': out.item()
        }


class AdaMR2DistGeneration(AdaMR):

    def __call__(self, k: int = 1) -> typing.Mapping:
        assert k <= 10
        tgt = self._tokenize(f"{self.tokenizer.CLS}{self.tokenizer.BOS}")
        smis, probs = [], []
        for _ in range(k):
            smi, prob = self._random_sample(tgt=tgt)
            smis.append(smi)
            probs.append(prob)
        return {
            'smiles': smis,
            'probabilities': probs
        }


class AdaMR2GoalGeneration(AdaMR):

    def __call__(self, goal: float, k: int = 1) -> typing.Mapping:
        assert k <= 10
        tgt = self._tokenize(f"{self.tokenizer.CLS}{self.tokenizer.BOS}")
        goal = torch.tensor(goal, device=self.device)
        smis, probs = [], []
        for _ in range(k):
            smi, prob = self._random_sample(goal=goal, tgt=tgt)
            smis.append(smi)
            probs.append(prob)
        return {
            'smiles': smis,
            'probabilities': probs
        }
