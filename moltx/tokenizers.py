import json
import typing
import re
import random
import os
from dataclasses import dataclass


class SmilesAtomwiseTokenizer:
    REGEX = r"(\[[^\]]+]|Br?|Cl?|N|O|S|P|F|I|b|c|n|o|s|p|\(|\)|\.|=|#|-|\+|\\|\/|:|~|@|\?|>|\*|\$|\%[0-9]{2}|[0-9])"

    def __init__(self, exclusive: typing.Optional[typing.Sequence[str]] = None) -> None:
        """
        Tokenize a SMILES molecule at atom-level:
            (1) 'Br' and 'Cl' are two-character tokens
            (2) Symbols with bracket are considered as tokens

        exclusive: A list of specifical symbols with bracket you want to keep. e.g., ['[C@@H]', '[nH]'].
        Other symbols with bracket will be replaced by '<unk>'. default is `None`.
        """
        self._regex = re.compile(self.REGEX)
        self._exclusive = exclusive

    def __call__(self, smiles: str) -> typing.Sequence[str]:
        tokens = self._regex.findall(smiles)
        if self._exclusive:
            for i, token in enumerate(tokens):
                if token.startswith('[') and token not in self._exclusive:
                    tokens[i] = '<unk>'
        return tokens


class SmilesTokenizer:
    """
    Tokenize SMILES based on the learned SPE tokens.

    codes: output file of `learn_SPE()`

    merges: number of learned SPE tokens you want to use. `-1` means using all of them. `1000` means use the most frequent 1000.

    exclusive_tokens: argument that passes to  `atomwise_tokenizer()`

    dropout: See [BPE-Dropout: Simple and Effective Subword Regularization](https://arxiv.org/abs/1910.13267).
    If `dropout` is set to 0, the segmentation is equivalent to the standard BPE; if `dropout` is set to 1, the segmentation splits words into distinct characters.
    """

    def __init__(self, codes_path: typing.Optional[str] = None, dropout: float = 0.0, merges: int = -1, exclusive_tokens: typing.Optional[typing.Sequence[str]] = None) -> None:
        self.smi_tkz = SmilesAtomwiseTokenizer(exclusive_tokens)
        self.dropout = dropout
        self.bpe_codes = {}
        if codes_path is not None:
            with open(codes_path) as f:
                bpe_codes = [tuple(item.strip().split(' ')) for (
                    n, item) in enumerate(f) if (n < merges or merges == -1)]
            for i, item in enumerate(bpe_codes):
                if len(item) != 2:
                    raise RuntimeError(f"Invalid BPE code at line: {i}")
            self.bpe_codes = dict([(code, i)
                                   for (i, code) in enumerate(bpe_codes)])

    def __call__(self, smiles: str) -> typing.Sequence[str]:
        if len(smiles) == 1:
            return [smiles]
        tokens = self.smi_tkz(smiles)
        if self.dropout >= 0.999999 or not self.bpe_codes:
            return tokens
        while len(tokens) > 1:
            pairs = [(self.bpe_codes[pair], i, pair) for (i, pair) in enumerate(zip(tokens, tokens[1:])) if (
                not self.dropout or random.random() > self.dropout) and pair in self.bpe_codes]
            if not pairs:
                break
            # get first merge operation in list of BPE codes
            bigram = min(pairs)[2]
            positions = [i for (rank, i, pair) in pairs if pair == bigram]
            i = 0
            new_tokens = []
            bigram = ''.join(bigram)
            for j in positions:
                # merges are invalid if they start before current position. This can happen if there are overlapping pairs: (x x x -> xx x)
                if j < i:
                    continue
                # all symbols before merged pair
                new_tokens.extend(tokens[i:j])
                new_tokens.append(bigram)  # merged pair
                i = j + 2  # continue after merged pair
            # add all symbols until end of tokens
            new_tokens.extend(tokens[i:])
            tokens = new_tokens
        return tokens


class NumericalTokenizer:
    REGEX = r"([+-]?\d|\.)"

    def __init__(self):
        self._regex = re.compile(self.REGEX)

    def __call__(self, number: str) -> typing.Sequence[str]:
        digits = self._regex.findall(number)
        try:
            dot = digits.index('.')
        except ValueError:
            dot = len(digits)
        tokens = digits.copy()
        for idx, v in enumerate(digits):
            if idx == dot:
                continue
            p = dot - idx
            if idx < dot:
                p -= 1
            t = f'_{v}_{p}_'
            tokens[idx] = t
        return tokens


@dataclass
class MoltxPretrainConfig:
    token_size: int
    fmt: str = 'smiles'
    data_dir: str = os.path.join(os.path.dirname(__file__), 'data')
    spe: bool = True
    spe_dropout: float = 0.0
    spe_merges: int = -1


class MoltxTokenizer:
    REGEX = re.compile(r"<\w{3}>")
    PAD = "<pad>"
    UNK = "<unk>"
    BOS = "<bos>"
    EOS = "<eos>"
    SEP = "<sep>"
    CLS = "<cls>"
    RESERVED = (PAD, UNK, BOS, EOS, SEP, CLS)

    @classmethod
    def from_pretrain(cls, conf: MoltxPretrainConfig) -> 'MoltxTokenizer':
        datadir = conf.data_dir
        kwargs = {
            'token_size': conf.token_size
        }
        if conf.spe:
            kwargs['spe_codes_path'] = os.path.join(datadir, f'spe_{conf.fmt}.txt')
            kwargs['spe_dropout'] = conf.spe_dropout
            kwargs['spe_merges'] = conf.spe_merges
        tkz = cls(**kwargs, freeze=True)
        tkz.load(os.path.join(datadir, f'tks_{conf.fmt}.json'))
        return tkz

    def __init__(self, token_size: int, freeze: bool = False, spe_codes_path: typing.Optional[str] = None, spe_dropout: float = 0.0, spe_merges: int = -1) -> None:
        self._tokens = []
        self._token_idx = {}
        self._token_size = token_size
        self._freeze = freeze
        self._update_tokens(self.RESERVED)
        spe_kwargs = {}
        if spe_codes_path is not None:
            spe_kwargs['codes_path'] = spe_codes_path
            spe_kwargs['merges'] = spe_merges
            spe_kwargs['dropout'] = spe_dropout
        self._smi_tkz = SmilesTokenizer(**spe_kwargs)

    def _update_tokens(self, tokens: typing.Sequence[str]) -> None:
        if self._freeze:
            return
        for token in tokens:
            if token not in self._token_idx:
                t = len(self._tokens)
                if t >= self._token_size:
                    return
                self._token_idx[token] = t
                self._tokens.append(token)

    def _load_tokens(self, tokens: typing.Sequence[str]) -> None:
        freeze = self._freeze
        self._freeze = False
        self._update_tokens(tokens)
        self._freeze = freeze

    def __getitem__(self, item: typing.Union[int, str]) -> str:
        if isinstance(item, int):
            try:
                return self._tokens[item]
            except IndexError:
                return self.UNK
        return self._token_idx.get(item, self._token_idx[self.UNK])

    def __len__(self) -> int:
        return len(self._tokens)

    def loads(self, tokens_json: str) -> 'MoltxTokenizer':
        tokens = json.loads(tokens_json)['tokens']
        return self._load_tokens(tokens)

    def load(self, path: str) -> 'MoltxTokenizer':
        with open(path, 'r') as f:
            return self.loads(f.read())

    def dumps(self) -> str:
        return json.dumps({
            'tokens': self._tokens
        })

    def dump(self, path: str) -> None:
        with open(path, 'w') as f:
            f.write(self.dumps())

    def smi2tokens(self, smiles: str) -> typing.Sequence[str]:
        tokens = []
        m = self.REGEX.search(smiles)
        pos = 0
        while m is not None:
            start, end = m.span()
            if start > pos:
                tokens.extend(self._smi_tkz(smiles[pos:start]))
            tokens.append(m[0])
            pos = end
            m = self.REGEX.search(smiles, pos=pos)
        if len(smiles) > pos:
            tokens.extend(self._smi_tkz(smiles[pos:]))
        self._update_tokens(tokens)
        return tokens

    def decode(self, token_idxs: typing.Sequence[int]) -> str:
        tokens = [self[idx] for idx in token_idxs]
        return ''.join(tokens)

    def encode(self, smiles: str) -> typing.Sequence[int]:
        tokens = self.smi2tokens(smiles)
        return [self[t] for t in tokens]

    def __call__(self, smiles: str) -> typing.Sequence[int]:
        return self.encode(smiles)
