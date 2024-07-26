import re
import random
import typing
import logging
import torch
import torch.nn as nn
from moltx import tokenizers, models
import datamol as dm
from contextlib import suppress
from collections import Counter
import safe as sf
from safe import converter


class _Base:
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


class _GenBase(_Base):
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


class AdaMR(_GenBase):
    def __init__(self, model: models.AdaMR, device: torch.device = torch.device('cpu')) -> None:
        tokenizer = tokenizers.MoltxTokenizer.from_pretrain(models.AdaMRTokenizerConfig.Spe)
        super().__init__(tokenizer, model, device)

    def _model_args(self, smiles: str) -> typing.Tuple[torch.Tensor]:
        src = self._tokenize(smiles)
        tgt = self._tokenize(self.tokenizer.BOS)
        return src, tgt

    def __call__(self, smiles: str = "") -> typing.Mapping:
        src, tgt = self._model_args(smiles)
        smi, prob = self._beam_search(src=src, tgt=tgt, beam_width=3)[0]
        return {
            'smiles': smi,
            'probability': prob
        }


class AdaMRClassifier(AdaMR):
    def _model_args(self, smiles: str) -> typing.Tuple[torch.Tensor]:
        src = self._tokenize(smiles)
        tgt = self._tokenize(self.tokenizer.CLS)
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
        tgt = self._tokenize(self.tokenizer.CLS)
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


class AdaMR2(_GenBase):
    def __init__(self, model: models.AdaMR2, device: torch.device = torch.device('cpu')) -> None:
        tokenizer = tokenizers.MoltxTokenizer.from_pretrain(models.AdaMRTokenizerConfig.Spe)
        super().__init__(tokenizer, model, device)

    def __call__(self, smiles: str = "") -> typing.Mapping:
        tgt = self._tokenize(f"{smiles}{self.tokenizer.BOS}")
        smi, prob = self._beam_search(tgt=tgt, beam_width=3)[0]
        return {
            'smiles': smi,
            'probability': prob
        }


class AdaMR2Classifier(AdaMR2):

    def __call__(self, smiles: str) -> typing.Mapping:
        tgt = self._tokenize(f"{smiles}{self.tokenizer.CLS}")
        out = self.model(tgt)
        prob, label = out.softmax(-1).max(-1)
        return {
            'label': label.item(),
            'probability': prob.item()
        }


class AdaMR2Regression(AdaMR2):

    def __call__(self, smiles: str) -> typing.Mapping:
        tgt = self._tokenize(f"{smiles}{self.tokenizer.CLS}")
        out = self.model(tgt)
        return {
            'value': out.item()
        }


class AdaMR2DistGeneration(AdaMR2):

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


class AdaMR2GoalGeneration(AdaMR2):

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


class AdaMR2SuperGeneration(AdaMR2):
    ALLOW_GEN_TYPE = (
        'linker_generation',
        'scaffold_morphing',
        'motif_extension',
        'super_structure',
        'scaffold_decoration',
        'denovo_generation',
    )

    def __init__(self, model: nn.Module, device: torch.device = torch.device('cpu')) -> None:
        super().__init__(model, device)
        self.safe_encoder = converter.SAFEConverter()

    def __call__(self, gen_type: str, **kwargs) -> typing.Sequence:
        if gen_type not in self.ALLOW_GEN_TYPE:
            raise ValueError(f'invalidate input gen type: {gen_type}')
        return getattr(self, f"_gen_{gen_type}")(**kwargs)

    def _gen_linker_generation(
        self,
        side_chains: typing.Sequence[str],
        n_samples_per_trial: int = 10,
        n_trials: int = 1,
        sanitize: bool = False,
        do_not_fragment_further: bool = True,
        random_seed: int = None,
        **kwargs,
    ) -> typing.Sequence:
        """
        Linker generation is really just scaffold morphing underlying.

        Args:
            side_chains: list of fragments to link together, they are joined in the order provided
            n_samples_per_trial: number of new molecules to generate for each randomization
            n_trials: number of randomization to perform
            do_not_fragment_further: whether to fragment the scaffold further or not
            sanitize: whether to sanitize the generated molecules
            random_seed: random seed to use
            kwargs: any argument to provide to the underlying generation function
        """

        if len(side_chains) != 2:
            raise ValueError(
                "Linker generation only works when providing two groups as side chains"
            )
        return self._fragment_linking(
            side_chains=side_chains,
            n_samples_per_trial=n_samples_per_trial,
            n_trials=n_trials,
            sanitize=sanitize,
            do_not_fragment_further=do_not_fragment_further,
            random_seed=random_seed,
            is_linking=True,
            **kwargs,
        )

    def _gen_scaffold_morphing(
        self,
        side_chains: typing.Union[str, typing.Sequence[str]] = None,
        mol: str = None,
        core: str = None,
        n_samples_per_trial: int = 10,
        n_trials: int = 1,
        sanitize: bool = False,
        do_not_fragment_further: bool = True,
        random_seed: int = None,
        **kwargs
    ) -> typing.Sequence:
        """
        For scaffold morphing, we try to replace the core by a new one. If the side_chains are provided, we use them.
        If a combination of molecule and core is provided, then, we use them to extract the side chains and performing the
        scaffold morphing then.

        !!! note "Finding the side chains"
            The algorithm to find the side chains from core assumes that the core we get as input has attachment points.
            Those attachment points are never considered as part of the query, rather they are used to define the attachment points.
            See ~sf.utils.compute_side_chains for more information.

        Args:
            side_chains: side chains to use to perform scaffold morphing (joining as best as possible the set of fragments)
            mol: input molecules when side_chains are not provided
            core: core to morph into another scaffold
            n_samples_per_trial: number of new molecules to generate for each randomization
            n_trials: number of randomization to perform
            do_not_fragment_further: whether to fragment the scaffold further or not
            sanitize: whether to sanitize the generated molecules
            random_seed: random seed to use
            kwargs: any argument to provide to the underlying generation function
        """

        return self._fragment_linking(
            side_chains=side_chains,
            mol=mol,
            core=core,
            n_samples_per_trial=n_samples_per_trial,
            n_trials=n_trials,
            sanitize=sanitize,
            do_not_fragment_further=do_not_fragment_further,
            random_seed=random_seed,
            is_linking=False,
            **kwargs,
        )

    def _gen_motif_extension(
        self,
        motif: str,
        n_samples_per_trial: int = 10,
        n_trials: int = 1,
        sanitize: bool = False,
        do_not_fragment_further: bool = False,
        random_seed: int = None,
        **kwargs
    ) -> typing.Sequence:
        """
        Motif extension is really just scaffold decoration underlying.

        Args:
            motif: scaffold (with attachment points) to decorate
            n_samples_per_trial: number of new molecules to generate for each randomization
            n_trials: number of randomization to perform
            do_not_fragment_further: whether to fragment the scaffold further or not
            sanitize: whether to sanitize the generated molecules and check
            random_seed: random seed to use
            kwargs: any argument to provide to the underlying generation function
        """

        return self._gen_scaffold_decoration(
            motif,
            n_samples_per_trial=n_samples_per_trial,
            n_trials=n_trials,
            sanitize=sanitize,
            do_not_fragment_further=do_not_fragment_further,
            random_seed=random_seed,
            add_dot=True,
            **kwargs,
        )

    def _gen_super_structure(
        self,
        core: str,
        n_samples_per_trial: int = 10,
        n_trials: int = 1,
        sanitize: bool = False,
        do_not_fragment_further: bool = False,
        random_seed: int = None,
        attachment_point_depth: int = None,
        **kwargs
    ) -> typing.Sequence:
        """
        To generate super-structure, we basically just create various attachment points to the input core,
        then perform scaffold decoration.

        Args:
            core: input substructure to use. We aim to generate super structures of this molecule
            n_samples_per_trial: number of new molecules to generate for each randomization
            n_trials: number of different attachment points to consider
            do_not_fragment_further: whether to fragment the scaffold further or not
            sanitize: whether to sanitize the generated molecules
            random_seed: random seed to use
            attachment_point_depth: depth of opening the attachment points.
                Increasing this, means you increase the number of substitution point to consider.
            kwargs: any argument to provide to the underlying generation function
        """

        core = dm.to_mol(core)
        cores = sf.utils.list_individual_attach_points(core, depth=attachment_point_depth)
        # get the fully open mol, everytime too.
        cores.append(dm.to_smiles(dm.reactions.open_attach_points(core)))
        cores = list(set(cores))
        rng = random.Random(random_seed)
        rng.shuffle(cores)
        # now also get the single openining of an attachment point
        total_sequences = []
        n_trials = n_trials or 1
        for _ in range(n_trials):
            core = cores[_ % len(cores)]
            try:
                out = self._completion(
                    fragment=core,
                    n_samples_per_trial=n_samples_per_trial,
                    n_trials=1,
                    do_not_fragment_further=do_not_fragment_further,
                    sanitize=sanitize,
                    random_seed=random_seed,
                    **kwargs,
                )
                total_sequences.extend(out)
            except Exception as e:
                logging.error(e)

        if sanitize:
            logging.info(
                f"After sanitization, {len(total_sequences)} / {n_samples_per_trial*n_trials} ({len(total_sequences)*100/(n_samples_per_trial*n_trials):.2f} %)  generated molecules are valid !"
            )
        return total_sequences

    def _gen_scaffold_decoration(
        self,
        scaffold: str,
        n_samples_per_trial: int = 10,
        n_trials: int = 1,
        do_not_fragment_further: bool = False,
        sanitize: bool = False,
        random_seed: int = None,
        add_dot=True,
        **kwargs
    ) -> typing.Sequence:
        """
        For scaffold decoration, we basically starts with a prefix with the attachment point.
        We first convert the prefix into valid safe string.

        Args:
            scaffold: scaffold (with attachment points) to decorate
            n_samples_per_trial: number of new molecules to generate for each randomization
            n_trials: number of randomization to perform
            do_not_fragment_further: whether to fragment the scaffold further or not
            sanitize: whether to sanitize the generated molecules and check if the scaffold is still present
            random_seed: random seed to use
            add_dot: whether to add a dot at the end of the fragments to signal to the model that we want to generate a distinct fragment.
            kwargs: any argument to provide to the underlying generation function
        """

        total_sequences = self._completion(
            fragment=scaffold,
            n_samples_per_trial=n_samples_per_trial,
            n_trials=n_trials,
            do_not_fragment_further=do_not_fragment_further,
            sanitize=sanitize,
            random_seed=random_seed,
            add_dot=add_dot,
        )

        # if we require sanitization
        # then we should filter out molecules that do not match the requested
        if sanitize:
            total_sequences = sf.utils.filter_by_substructure_constraints(total_sequences, scaffold)
            logging.info(
                f"After sanitization, {len(total_sequences)} / {n_samples_per_trial*n_trials} ({len(total_sequences)*100/(n_samples_per_trial*n_trials):.2f} %)  generated molecules are valid !"
            )
        return total_sequences

    def _gen_denovo_generation(
        self,
        n_samples_per_trial: int = 10,
        sanitize: bool = False,
        n_trials: int = 1,
        **kwargs,
    ) -> typing.Sequence:
        """
        De novo generation is equivalent to not having any prefix.

        Args:
            n_samples_per_trial: number of new molecules to generate
            sanitize: whether to perform sanitization, aka, perform control to ensure what is asked is what is returned
            n_trials: number of randomization to perform
            kwargs: any argument to provide to the underlying generation function
        """

        total_sequences = []
        n_trials = n_trials
        for _ in range(n_trials):
            sequences = self._generate(k=n_samples_per_trial)['smiles']
            total_sequences.extend(sequences)
        total_sequences = self._decode_safe(
            total_sequences, canonical=True, remove_invalid=sanitize
        )

        if sanitize:
            logging.info(
                f"After sanitization, {len(total_sequences)} / {n_samples_per_trial*n_trials} ({len(total_sequences)*100/(n_samples_per_trial*n_trials):.2f} %) generated molecules are valid !"
            )
        return total_sequences

    def _generate(self, k: int = 1, smi_str: str = '') -> typing.Sequence:
        tgt = self._tokenize(f"{self.tokenizer.BOS}{smi_str}")
        smis, probs = [], []
        for _ in range(k):
            smi, prob = self._random_sample(tgt=tgt)
            smis.append(smi)
            probs.append(prob)
        return {
            'smiles': smis,
            'probabilities': probs
        }

    def _fragment_linking(
        self,
        side_chains: str = None,
        mol: str = None,
        core: str = None,
        n_samples_per_trial: int = 10,
        n_trials: int = 1,
        sanitize: bool = False,
        do_not_fragment_further: bool = False,
        random_seed: int = None,
        is_linking: bool = False,
        **kwargs
    ) -> typing.Sequence:
        """
        !!! note "Finding the side chains"
            The algorithm to find the side chains from core assumes that the core we get as input has attachment points.
            Those attachment points are never considered as part of the query, rather they are used to define the attachment points.
            See ~sf.utils.compute_side_chains for more information.

        Args:
            side_chains: side chains to use to perform scaffold morphing (joining as best as possible the set of fragments)
            mol: input molecules when side_chains are not provided
            core: core to morph into another scaffold
            n_samples_per_trial: number of new molecules to generate for each randomization
            n_trials: number of randomization to perform
            do_not_fragment_further: whether to fragment the scaffold further or not
            sanitize: whether to sanitize the generated molecules
            random_seed: random seed to use
            is_linking: whether it's a linking task or not.
                For linking tasks, we use a different custom strategy of completing up to the attachment signal
            kwargs: any argument to provide to the underlying generation function
        """

        if side_chains is None:
            if mol is None and core is None:
                raise ValueError(
                    "Either side_chains OR mol+core should be provided for scaffold morphing"
                )
            side_chains = sf.trainer.utils.compute_side_chains(mol, core)
        side_chains = (
            [dm.to_mol(x) for x in side_chains]
            if isinstance(side_chains, list)
            else [dm.to_mol(side_chains)]
        )
        side_chains = ".".join([dm.to_smiles(x) for x in side_chains])
        if "*" not in side_chains:
            logging.warning(
                f"Side chain {side_chains} does not contain any dummy atoms, this might not be what you want"
            )

        rng = random.Random(random_seed)
        new_seed = rng.randint(1, 1000)

        total_sequences = []
        n_trials = n_trials
        for _ in range(n_trials):
            with dm.without_rdkit_log():
                context_mng = (
                    sf.utils.attr_as(self.safe_encoder, "slicer", None)
                    if do_not_fragment_further
                    else suppress()
                )
                old_slicer = getattr(self.safe_encoder, "slicer", None)
                with context_mng:
                    try:
                        encoded_fragment = self.safe_encoder.encoder(
                            side_chains,
                            canonical=False,
                            randomize=False,
                            constraints=None,
                            allow_empty=True,
                            seed=new_seed,
                        )

                    except Exception as e:
                        logging.error(e)
                        raise sf.SAFEEncodeError(f"Failed to encode {side_chains}") from e
                    finally:
                        if old_slicer is not None:
                            self.safe_encoder.slicer = old_slicer

            fragments = encoded_fragment.split(".")

            missing_closure = Counter(self.safe_encoder._find_branch_number(encoded_fragment))
            missing_closure = [f"{str(x)}" for x in missing_closure if missing_closure[x] % 2 == 1]

            closure_pos = [
                m.start() for x in missing_closure for m in re.finditer(x, encoded_fragment)
            ]
            fragment_pos = [m.start() for m in re.finditer(r"\.", encoded_fragment)]
            min_pos = 0
            while fragment_pos[min_pos] < closure_pos[0] and min_pos < len(fragment_pos):
                min_pos += 1
            min_pos += 1
            max_pos = len(fragment_pos)
            while fragment_pos[max_pos - 1] > closure_pos[-1] and max_pos > 0:
                max_pos -= 1

            split_index = rng.randint(min_pos, max_pos)
            prefix, suffixes = ".".join(fragments[:split_index]), ".".join(fragments[split_index:])

            missing_prefix_closure = Counter(self.safe_encoder._find_branch_number(prefix))
            missing_suffix_closure = Counter(self.safe_encoder._find_branch_number(suffixes))
            missing_prefix_closure = (
                ["."] + [x for x in missing_closure if int(x) not in missing_prefix_closure] + ["."]
            )
            missing_suffix_closure = (
                ["."] + [x for x in missing_closure if int(x) not in missing_suffix_closure] + ["."]
            )

            mol_linker_slicer = sf.utils.MolSlicer(
                shortest_linker=(not is_linking), require_ring_system=(not is_linking)
            )
            prefix_smiles = converter.decode(prefix, remove_dummies=False, as_mol=False)
            suffix_smiles = converter.decode(suffixes, remove_dummies=False, as_mol=False)

            prefix_sequences = self._generate(k=n_trials, smi_str=prefix + ".")['smiles']
            suffix_sequences = self._generate(k=n_trials, smi_str=suffixes + ".")['smiles']
            prefix_sequences = self._decode_safe(
                prefix_sequences, canonical=True, remove_invalid=True
            )
            suffix_sequences = self._decode_safe(
                suffix_sequences, canonical=True, remove_invalid=True
            )
            sequences = self._mix_sequences(
                prefix_sequences,
                suffix_sequences,
                prefix_smiles,
                suffix_smiles,
                n_samples_per_trial,
                mol_linker_slicer,
            )
            total_sequences.extend(sequences)
        if sanitize:
            total_sequences = sf.utils.filter_by_substructure_constraints(
                total_sequences, side_chains
            )
            logging.info(
                f"After sanitization, {len(total_sequences)} / {n_samples_per_trial*n_trials} ({len(total_sequences)*100/(n_samples_per_trial*n_trials):.2f} %)  generated molecules are valid !"
            )
        return total_sequences

    def _completion(
        self,
        fragment: str,
        n_samples_per_trial: int = 10,
        n_trials: int = 1,
        do_not_fragment_further: bool = False,
        sanitize: bool = False,
        random_seed: int = None,
        add_dot: bool = False,
        is_safe: bool = False,
        **kwargs
    ):
        """Perform sentence completion using a prefix fragment

        Args:
            scaffold: scaffold (with attachment points) to decorate
            n_samples_per_trial: number of new molecules to generate for each randomization
            n_trials: number of randomization to perform
            do_not_fragment_further: whether to fragment the scaffold further or not
            sanitize: whether to sanitize the generated molecules
            random_seed: random seed to use
            is_safe: whether the smiles is already encoded as a safe string
            add_dot: whether to add a dot at the end of the fragments to signal to the model that we want to generate a distinct fragment.
            kwargs: any argument to provide to the underlying generation function
        """

        # Step 1: we conver the fragment into the relevant safe string format
        # we use the provided safe encoder with the slicer that was expected

        rng = random.Random(random_seed)
        new_seed = rng.randint(1, 1000)

        total_sequences = []
        n_trials = n_trials or 1
        for _ in range(n_trials):
            if is_safe:
                encoded_fragment = fragment
            else:
                with dm.without_rdkit_log():
                    context_mng = (
                        sf.utils.attr_as(self.safe_encoder, "slicer", None)
                        if do_not_fragment_further
                        else suppress()
                    )
                    old_slicer = getattr(self.safe_encoder, "slicer", None)
                    with context_mng:
                        try:
                            encoded_fragment = self.safe_encoder.encoder(
                                fragment,
                                canonical=False,
                                randomize=True,
                                constraints=None,
                                allow_empty=True,
                                seed=new_seed,
                            )

                        except Exception as e:
                            logging.error(e)
                            raise sf.SAFEEncodeError(f"Failed to encode {fragment}") from e
                        finally:
                            if old_slicer is not None:
                                self.safe_encoder.slicer = old_slicer

            if add_dot and encoded_fragment.count("(") == encoded_fragment.count(")"):
                encoded_fragment = encoded_fragment.rstrip(".") + "."

            sequences = self._generate(
                k=n_samples_per_trial, smi_str=encoded_fragment
            )['smiles']

            sequences = self._decode_safe(sequences, canonical=True, remove_invalid=sanitize)
            total_sequences.extend(sequences)

        return total_sequences

    def _decode_safe(
        self, sequences: typing.Sequence[str], canonical: bool = True, remove_invalid: bool = False
    ) -> typing.Sequence:
        """Decode a safe sequence into a molecule

        Args:
            sequence: safe sequence to decode
            canonical: whether to return canonical sequence
            remove_invalid: whether to remove invalid safe strings or keep them
        """

        def _decode_fn(x):
            return converter.decode(
                x,
                as_mol=False,
                fix=True,
                remove_added_hs=True,
                canonical=canonical,
                ignore_errors=True,
                remove_dummies=True,
            )

        if len(sequences) > 100:
            safe_strings = dm.parallelized(_decode_fn, sequences, n_jobs=-1)
        else:
            safe_strings = [_decode_fn(x) for x in sequences]
        if remove_invalid:
            safe_strings = [x for x in safe_strings if x is not None]

        return safe_strings

    def _mix_sequences(
        self,
        prefix_sequences: typing.Sequence[str],
        suffix_sequences: typing.Sequence[str],
        prefix: str,
        suffix: str,
        n_samples: int,
        mol_linker_slicer,
    ) -> typing.Sequence:
        """Use generated prefix and suffix sequences to form new molecules
        that will be the merging of both. This is the two step scaffold morphing and linker generation scheme
        Args:
            prefix_sequences: list of prefix sequences
            suffix_sequences: list of suffix sequences
            prefix: decoded smiles of the prefix
            suffix: decoded smiles of the suffix
            n_samples: number of samples to generate
        """

        prefix_linkers = []
        suffix_linkers = []
        prefix_query = dm.from_smarts(prefix)
        suffix_query = dm.from_smarts(suffix)

        for x in prefix_sequences:
            with suppress(Exception):
                x = dm.to_mol(x)
                out = mol_linker_slicer(x, prefix_query)
                prefix_linkers.append(out[1])
        for x in suffix_sequences:
            with suppress(Exception):
                x = dm.to_mol(x)
                out = mol_linker_slicer(x, suffix_query)
                suffix_linkers.append(out[1])
        n_linked = 0
        linked = []
        linkers = prefix_linkers + suffix_linkers
        linkers = [x for x in linkers if x is not None]
        for n_linked, linker in enumerate(linkers):
            linked.extend(mol_linker_slicer.link_fragments(linker, prefix, suffix))
            if n_linked > n_samples:
                break
            linked = [x for x in linked if x]
        return linked[:n_samples]
