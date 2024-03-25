import random
from typing import Any, Iterator, Callable, List, Tuple
from enum import Enum

import numpy as np

from ..op_graph import OpGraph, OpId
from ..dim_graph import DimGraph
from ..simulator import SimResult
from ..operators import FissionOp
from .rewrite_rules import resolve
from ..utils import LOG
from .. import utils
from . import misc


class GraphMutator:
    def __init__(self) -> None:
        pass

    def chain(self, *mutators):
        return ChainedMutator(self, *mutators)

    def zip(self, *mutators):
        return ZippedMutator(self, *mutators)

    def truncate(self, count):
        return TruncatedMutator(self, count)

    def random(self, count):
        return RandomMutator(self, count)

    def _gen_mutations(
        self, G: OpGraph, **kwargs
    ) -> Iterator[Tuple[Callable[[], OpGraph], Any]]:
        raise NotImplementedError()

    def gen_mutations(
        self, G: OpGraph, **kwargs
    ) -> Iterator[Tuple[Callable[[], OpGraph], Any]]:
        return self._gen_mutations(G, **kwargs)

    def __call__(
        self, G: OpGraph, **kwargs
    ) -> Iterator[Tuple[Callable[[], OpGraph], Any]]:
        return self.gen_mutations(G, **kwargs)


class CompoundMutator(GraphMutator):
    pass


class AtomMutator(GraphMutator):
    def gen_mutations(
        self, G: OpGraph, **kwargs
    ) -> Iterator[Tuple[Callable[[], OpGraph], Any]]:
        cnt = 0
        try:
            for res in self._gen_mutations(G, **kwargs):
                yield res
                cnt += 1
        finally:
            LOG.info(f"{self.__class__} generates {cnt} mutations")


class TruncatedMutator(CompoundMutator):
    def __init__(self, mutator: GraphMutator, count) -> None:
        super().__init__()
        self._mutator = mutator
        self._count = count

    def _gen_mutations(
        self, G: OpGraph, **kwargs
    ) -> Iterator[Tuple[Callable[[], OpGraph], Any]]:
        i = 0
        gen = self._mutator.gen_mutations(G, **kwargs)
        try:
            for res in gen:
                if i >= self._count:
                    break
                i += 1
                yield res
        finally:
            gen.close()


class RandomMutator(CompoundMutator):
    def __init__(self, mutator: GraphMutator, count) -> None:
        super().__init__()
        self._mutator = mutator
        self._count = count

    def _gen_mutations(
        self, G: OpGraph, **kwargs
    ) -> Iterator[Tuple[Callable[[], OpGraph], Any]]:
        collected = list(self._mutator.gen_mutations(G, **kwargs))
        collected = random.choices(collected, k=self._count)
        for res in collected:
            yield res


class ChainedMutator(CompoundMutator):
    def __init__(self, *mutators) -> None:
        super().__init__()
        self._mutators: List[GraphMutator] = mutators

    def _gen_mutations(
        self, G: OpGraph, **kwargs
    ) -> Iterator[Tuple[Callable[[], OpGraph], Any]]:
        for m in self._mutators:
            gen = m.gen_mutations(G, **kwargs)
            try:
                yield from gen
            finally:
                gen.close()


class ZippedMutator(CompoundMutator):
    def __init__(self, *mutators) -> None:
        super().__init__()
        self._mutators: List[GraphMutator] = mutators

    def _gen_mutations(
        self, G: OpGraph, **kwargs
    ) -> Iterator[Tuple[Callable[[], OpGraph], Any]]:
        gens = [m.gen_mutations(G, **kwargs) for m in self._mutators]
        stopped = False
        try:
            while not stopped:
                stopped = True
                for it in gens:
                    res = next(it, None)
                    if res is not None:
                        stopped = False
                        yield res
        finally:
            for gen in gens:
                gen.close()


class RewriteRuleMutator(AtomMutator):
    TASO_RULES = [
        *(f"AddMulRule{i}" for i in range(27)),
        *(
            f"{pre}{mid}Rule{i}"
            for pre in ("Concat", "Merge")
            for mid in ("Matmul", "Conv2d")
            for i in ("", "1")
        ),
        "MergeFlexMatmulRule",
    ]
    MEM_RULES = ["SwappingRule", "RecomputeRule1", "RecomputeRule2"]
    DEMEM_RULES = [
        "DeSwappingRule1",
        "DeSwappingRule2",
        "DeSwappingRule3",
        "DeRecomputeRule1",
        "DeRecomputeRule2",
    ]

    def __init__(self, rules="taso", addition_rules=None) -> None:
        super().__init__()
        if isinstance(rules, str):
            rules = [
                r
                for rr in rules.split(",")
                for r in getattr(self, f"{rr.upper()}_RULES")
            ]
        self._rules = [resolve(r) for r in rules]
        if addition_rules is not None:
            self._rules.extend(resolve(r) for r in addition_rules)

    def _gen_mutations(
        self, G: OpGraph, **kwargs
    ) -> Iterator[Tuple[Callable[[], OpGraph], Any]]:
        for rule in self._rules:
            for fs_graph, lv, _ in G.iter_fs_graphs(order="post"):
                for ms in rule.match(fs_graph):
                    yield (lambda lv, ms: lambda: G.fs_set_graph(lv, ms.apply()))(
                        lv, ms
                    ), f"{rule.__class__.__name__} {sorted(ms.g2p_maps.keys())}"


class SwapRematMutator(AtomMutator):
    def _gen_mutations(
        self, G: OpGraph, sim_res: SimResult = None, **kwargs
    ) -> Iterator[Callable[[], OpGraph]]:
        for v, ws, _ in self.select_swap_remat_candidates(sim_res):
        # for v, ws, _ in sim_res.select_swap_remat_candidates():
            yield (lambda v, ws: lambda: misc.apply_swap(G, v, ws))(
                v, ws
            ), f"swap {v} {ws}"
            if len(G.fs_pre_ids(v)) > 0:
                yield (lambda v, ws: lambda: misc.apply_remat(G, v, ws))(
                    v, ws
                ), f"remat {v} {ws}"

    @staticmethod
    def select_swap_remat_candidates(
        sim_res: SimResult,
    ) -> List[Tuple[OpId, OpId, float]]:
        peak_t = sim_res.mem_usage.peak_timept
        candidates = []
        for v in sim_res.mem_usage.peak_op_ids:
            op = sim_res.graph.fs_get_op(v)
            if op.is_mem_op():
                continue
            v_life = sim_res.op_lifes[v]
            assert v_life.live_in(peak_t)
            if v_life.exec_in(peak_t):
                continue
            ws_lifes_ = [(w, sim_res.op_lifes[w]) for w in sim_res.graph.fs_suc_ids(v)]
            ws_lifes = [
                (w, life)
                for w, life in ws_lifes_
                if life.start_time > peak_t
                and not sim_res.graph.fs_get_op(w).is_mem_op()
            ]
            if len(ws_lifes) <= 0 or len(ws_lifes) == len(ws_lifes_):
                continue
            score = (
                min(
                    min(life.start_time for _, life in ws_lifes) - peak_t,
                    peak_t - v_life.end_time,
                )
                * 1
            )
            candidates.append((v, [w for w, _ in ws_lifes], score))
        return sorted(candidates, key=lambda t: t[-1], reverse=True)


class FissionFactorMutator(AtomMutator):
    def __init__(self, directions=(-1, 1)) -> None:
        super().__init__()
        self._directions = directions

    def _gen_mutations(
        self, G: OpGraph, **kwargs
    ) -> Iterator[Tuple[Callable[[], OpGraph], Any]]:
        for F, lv, fs_op in G.iter_fs_graphs():
            if fs_op is None:
                continue
            factors = utils.factors(fs_op.fs_length)
            cur_idx = factors.index(fs_op.fs_factor)
            for d in self._directions:
                new_idx = cur_idx + d
                if not (0 <= new_idx < len(factors)):
                    continue
                new_factor = factors[new_idx]

                def _closure(lv, fs_op, new_factor):
                    return lambda: G.fs_set_op(
                        lv, misc.update_fs_factor(fs_op, new_factor, inplace=False)
                    )

                yield (
                    _closure(lv, fs_op, new_factor),
                    f"fission factor {lv} {fs_op.fs_factor} {new_factor}",
                )


class NaiveFissionTreeMutator(AtomMutator):
    def __init__(self) -> None:
        super().__init__()

    def _gen_mutations(
        self, G: OpGraph, **kwargs
    ) -> Iterator[Tuple[Callable[[], OpGraph], Any]]:
        while G.fs_height() >= 1:
            G = misc.remove_deepest_fission(G, inplace=False)
        topo_order = {id_: i for i, id_ in enumerate(G.fwd_rpo_topo_order)}
        D = G.dim_graph

        for D_sub_ids in sorted(D.connected_components(), key=len):
            D_sub_ids = sorted(D_sub_ids, key=lambda did: topo_order[D[did].op_id])

            for span in range(2, len(D_sub_ids) + 1):
                for start in range(0, len(D_sub_ids) + 1 - span):
                    end = start + span
                    D_sub = D.subgraph(D_sub_ids[start:end])
                    if D_sub.is_connected():
                        yield (
                            lambda D_sub: lambda: misc.apply_fission(
                                fs_nparts=2, dim_graph=D_sub, inplace=False
                            )
                        )(D_sub), "naive fission tree"


class FissionTreeMutator(AtomMutator):
    def __init__(self, max_level=4, height_limit=1, heat_policy=2) -> None:
        super().__init__()
        self._max_level = max(1, max_level)
        self._height_limit = max(1, height_limit)
        assert 0 <= heat_policy < 5
        self._heat_policy = heat_policy

    def _gen_mutations(
        self, G: OpGraph, sim_res: SimResult = None, **kwargs
    ) -> Iterator[Tuple[Callable[[], OpGraph], Any]]:
        _fs_tree_level_attr = "_fs_tree_level"
        level = getattr(G, _fs_tree_level_attr, None)
        if level is None:
            level = self._max_level
        if level <= 0:
            return

        D = G.dim_graph
        D_subs = D.connected_components()
        max_len = max(map(len, D_subs))
        for D_sub in D_subs:
            if len(D_sub) <= max_len / 2:
                continue
            
            D_sub = D.subgraph(D_sub)

            dim_len = D_sub.get_connected_dim_len()
            factors = utils.factors(dim_len)
            if len(factors) < 2:
                return
            new_dim_len = min([factors[-2], *(op.fs_factor for _, op in G.sub_fs_ops)])

            if self._heat_policy == 4:
                assert sim_res is not None
                mem_heats = np.zeros([G.max_id() + 1], dtype="float32")
                for v in sim_res.mem_usage.peak_op_ids:
                    mem_heats[utils.to_tuple(v)[0]] += G.fs_get_op(v).out_memory
            else:
                mem_heats = G.get_memory_heats(self._heat_policy)

            scale = new_dim_len / dim_len
            new_mem_heats = mem_heats * scale
            new_mem_heats[G.sub_fs_ids] /= scale

            dom_table = D_sub.op_dom_table
            peak_mem = mem_heats.sum()
            peak_mems = dom_table.T @ mem_heats
            new_peak_mems = dom_table.T @ new_mem_heats + mem_heats * (1 - scale)
            score_board = (peak_mems - new_peak_mems) / peak_mem
            scores, dom_ids = zip(
                *sorted(
                    (s, v)
                    for v, s in enumerate(score_board)
                    if s > 0 and not G[v].is_fission()
                )
            )
            if len(scores) <= 0:
                setattr(G, _fs_tree_level_attr, 0)
                return
            hi = scores[-1]
            lo = 0
            pivot_scores = [lo + (hi - lo) * (n / level) for n in range(1, level + 1)]
            pivots = sorted(set(np.searchsorted(scores, pivot_scores))) + [len(scores)]
            level = len(pivots) - 1
            if level == 0:
                setattr(G, _fs_tree_level_attr, 0)
                return
            dom_ids = sorted(
                dom_ids[slice(*pivots[:2])], key=lambda v: len(dom_table[:, v])
            )
            candidates = []
            for v in dom_ids:
                if dom_table[candidates, v].any():
                    continue
                candidates.append(v)

            def _closure(dom_id):
                def _thunk():
                    G = misc.apply_fission(
                        dim_graph=D_sub,
                        dom_id=dom_id,
                        fs_factor=new_dim_len,
                        inplace=False,
                    )
                    if G.fs_height() > self._height_limit:
                        G = misc.remove_deepest_fission(G, inplace=True)
                    setattr(G, _fs_tree_level_attr, level - 1)
                    return G

                return _thunk

            for dom_id in candidates:
                yield _closure(dom_id), f"fission tree {dom_id}"
