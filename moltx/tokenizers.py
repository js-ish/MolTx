import json
import torch
import typing
import re
import random
import os


class SmilesAtomwiseTokenizer:
    REGEX = r"(\[[^\]]+]|Br?|Cl?|N|O|S|P|F|I|b|c|n|o|s|p|\(|\)|\.|=|#|-|\+|\\|\/|:|~|@|\?|>|\*|\$|\%[0-9]{2}|[0-9])"

    def __init__(self, exclusive: typing.Optional[typing.List[str]] = None) -> None:
        """
        Tokenize a SMILES molecule at atom-level:
            (1) 'Br' and 'Cl' are two-character tokens
            (2) Symbols with bracket are considered as tokens

        exclusive: A list of specifical symbols with bracket you want to keep. e.g., ['[C@@H]', '[nH]'].
        Other symbols with bracket will be replaced by '<unk>'. default is `None`.
        """
        self._regex = re.compile(self.REGEX)
        self._exclusive = exclusive

    def __call__(self, smiles: str) -> typing.List[str]:
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

    def __init__(self, codes_path: typing.Optional[str] = None, dropout: float = 0.0, merges: int = -1, exclusive_tokens: typing.Optional[typing.List[str]] = None) -> None:
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

    def __call__(self, smiles: str) -> typing.List[str]:
        if len(smiles) == 1:
            return [smiles]
        tokens = self.smi_tkz(smiles)
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

    def __call__(self, number: str) -> typing.List[str]:
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


class MoltxTokenizer:
    def __init__(self, token_size: int = 512, seq_len: int = 256, freeze: bool = False, dropout: float = 1.000000001, spe_codes: typing.Optional[str] = None, spe_merges: int = -1, molecule_type : str = 'smiles') -> None:
        self._tokens = []
        self._token_idx = {}
        self._token_size = token_size
        self._seq_len = seq_len
        self._freeze = freeze
        self._update_tokens(self.reserved)
        spe_kwargs = {'dropout': dropout}
        if spe_codes is not None:
            spe_kwargs['codes_path'] = spe_codes
            spe_kwargs['merges'] = spe_merges
        self._smi_tkz = SmilesTokenizer(**spe_kwargs)

    def _update_tokens(self, tokens: typing.List[str]) -> None:
        if self._freeze:
            return
        for token in tokens:
            if token not in self._token_idx:
                l = len(self._tokens)
                if l >= self._token_size:
                    return
                self._token_idx[token] = l
                self._tokens.append(token)

    def _load_tokens(self, tokens: typing.List[str]) -> None:
        freeze = self._freeze
        self._freeze = False
        self._update_tokens(tokens)
        self._freeze = freeze

    def __call__(self, smiles: typing.Union[str, typing.List[int]], seqlen: int = 0,
                 add_bos: bool = False, add_eos: bool = False, add_cls: bool = False,
                 multi_input: bool = False, padding: bool = True) -> torch.Tensor:
        if seqlen <= 0:
            seqlen = self._seq_len
        if isinstance(smiles, str):
            smiles = self.encode(smiles, maxlen=seqlen)
        smiles = smiles[:seqlen-1]
        if add_bos:
            smiles = [self._token_idx[self.bos]] + smiles
        if add_eos:
            smiles = smiles + [self._token_idx[self.eos]]
        if add_cls:
            smiles = [self._token_idx[self.cls]] + smiles
        tokens_ = torch.zeros(seqlen, dtype=torch.int)
        for i, token in enumerate(smiles):
            tokens_[i] = token
        if not padding:
            last_char = min((tokens_ == self._token_idx[self.pad]).nonzero())
            tokens_ = tokens_[:last_char]
        return tokens_

    def __getitem__(self, item: typing.Union[int, str]) -> str:
        if isinstance(item, int):
            return self._tokens[item]
        return self._token_idx.get(item, self._token_idx[self.unk])

    def __len__(self) -> int:
        return len(self._tokens)

    @property
    def seq_len(self) -> int:
        return self._seq_len

    @classmethod
    def from_jsonfile(cls, molecule_type: str = 'smiles', *args, **kwargs) -> 'MoltxTokenizer':
        kwargs['spe_codes'] = os.path.join(os.path.dirname(__file__), 'data', f'spe_{molecule_type}.txt')
        tkz = cls(*args, **kwargs, freeze=True)
        tkz.load(os.path.join(os.path.dirname(__file__), 'data', f'tks_{molecule_type}.json'))
        return tkz

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

    def encode(self, smiles: str, maxlen: typing.Optional[int] = None) -> typing.List[int]:
        tokens = self._smi_tkz(smiles)
        if maxlen is not None and len(tokens) >= maxlen:
            return []
        return [self._token_idx.get(t, self._token_idx[self.unk]) for t in tokens]

    def decode(self, tokens: typing.List[int]) -> str:
        eos = self._token_idx[self.eos]
        pad = self._token_idx[self.pad]
        bos = self._token_idx[self.bos]
        smi = []
        start_idx = 1 if tokens[0] == bos else 0
        token_len = len(self._tokens)
        for i in tokens[start_idx:]:
            if i == eos or i == pad:
                break
            if i < token_len:
                smi.append(self._tokens[i])
        return ''.join(smi)

    @property
    def pad(self):
        return '<pad>'

    @property
    def unk(self):
        return '<unk>'

    @property
    def bos(self):
        return '<bos>'

    @property
    def eos(self):
        return '<eos>'

    @property
    def sep(self):
        return '<sep>'

    @property
    def cls(self):
        return '<cls>'

    @property
    def reserved(self):
        return (self.pad, self.unk, self.bos, self.eos, self.sep, self.cls)
