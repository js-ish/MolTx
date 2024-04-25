import os.path
from moltx import tokenizers as tkz


def test_smi_tkz():

    smi = 'CC[N+](C)(C)Cc1ccccc1Br'

    tok = tkz.SmilesAtomwiseTokenizer()

    tokens = tok(smi)
    assert tokens == ['C', 'C', '[N+]', '(', 'C', ')', '(', 'C', ')',
                      'C', 'c', '1', 'c', 'c', 'c', 'c', 'c', '1', 'Br']


def test_safe_tkz():

    safe = 'N18CC[C@H]CC1.O=C6C#CC8.N67.c17ccc2ncnc4c2c1.N45.c15cccc(Br)c1'

    tok = tkz.SmilesAtomwiseTokenizer()

    tokens = tok(safe)
    assert tokens == ['N', '1', '8', 'C', 'C', '[C@H]', 'C', 'C', '1', '.', 'O', '=', 'C', '6', 'C', '#', 'C', 'C', '8', '.', 'N', '6', '7', '.', 'c', '1', '7',
                      'c', 'c', 'c', '2', 'n', 'c', 'n', 'c', '4', 'c', '2', 'c', '1', '.', 'N', '4', '5', '.', 'c', '1', '5', 'c', 'c', 'c', 'c', '(', 'Br', ')', 'c', '1']


def test_num_tkz():
    tok = tkz.NumericalTokenizer()
    number = '123.659'
    tokens = tok(number)

    assert tokens == ['_1_2_', '_2_1_', '_3_0_',
                      '.', '_6_-1_', '_5_-2_', '_9_-3_']


def test_spe_tkz(datadir):
    smi = "CC[N+](C)(C)Cc1ccccc1Br"
    tok = tkz.SmilesTokenizer(codes_path=os.path.join(datadir, 'spe_smiles.txt'))
    tokens = tok(smi)
    assert tokens == ['CC', '[N+](C)', '(C)C', 'c1ccccc1', 'Br']
    assert tok('C') == ['C']


def test_spe_safe_tkz(datadir):
    smi = "N18CC[C@H]CC1.O=C6C#CC8.N67.c17ccc2ncnc4c2c1.N45.c15cccc(Br)c1"
    tok = tkz.SmilesTokenizer(codes_path=os.path.join(datadir, 'spe_safe.txt'))
    tokens = tok(smi)
    assert tokens == ['N18', 'CC[C@H]', 'CC1', '.O=C', '6C', '#', 'CC8',
                      '.N67', '.c17', 'ccc2', 'nc', 'nc4', 'c2c1', '.N45', '.c15', 'cccc(Br)c1']
    assert tok('C') == ['C']


def test_moltx_tkz():
    tok = tkz.MoltxTokenizer(token_size=128)

    tokens = tok.smi2tokens('<pad>')
    assert tokens == ['<pad>']

    tokens = tok.smi2tokens('c1ccccc1<sep><cls>c1ccccc1')
    assert tokens == ['c', '1', 'c', 'c', 'c', 'c', 'c', '1',
                      '<sep>', '<cls>', 'c', '1', 'c', 'c', 'c', 'c', 'c', '1']

    tokens = tok('BrCl<pad>')
    assert tokens[2] == 0
    assert tok.decode(tokens) == 'BrCl<pad>'

    datadir = '/tmp'
    tok.dump(os.path.join(datadir, 'tks_smiles.json'))
    conf = tkz.MoltxPretrainConfig(token_size=128, spe=False, data_dir=datadir)
    tok2 = tkz.MoltxTokenizer.from_pretrain(conf=conf)
    assert tok._tokens == tok2._tokens
    assert tok._token_idx == tok2._token_idx

    stok = tkz.MoltxTokenizer(token_size=2)
    assert stok._token_size == 2
    assert len(stok._tokens) == 2
    tokens = stok('BrCl')
    assert tokens[0] == 1 # <unk>
