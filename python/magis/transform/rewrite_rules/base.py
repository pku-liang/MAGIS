from dataclasses import dataclass
from typing import List, Dict, Tuple, Iterator, Union

import rustworkx as rx

from ...op_graph import OpGraph, Operator, OpId, ArgIndex


@dataclass
class MatchResult:
    ori_graph: OpGraph
    pat_subgraph: OpGraph
    g2p_maps: Dict[OpId, OpId]  # from ori_graph to pat_subgraph
    p2g_maps: Dict[OpId, OpId]  # from pat_subgraph to ori_graph
    g_inp_ids: List[OpId]
    g_out_ids: List[OpId]
    new_subgraph: OpGraph = None
    inp_maps: Dict[OpId, OpId] = None  # from ori_graph to new_subgraph
    out_maps: Dict[OpId, OpId] = None  # from ori_graph to new_subgraph

    def apply(
        self, inplace=False, return_id_maps=False
    ) -> Union[OpGraph, Tuple[OpGraph, Dict[int, int]]]:
        return self.ori_graph.replace_subgraph(
            set(self.g2p_maps.keys()),
            self.new_subgraph,
            self.inp_maps,
            self.out_maps,
            return_id_maps,
            inplace=inplace,
        )


def match_subgraph(
    self: OpGraph,
    pattern: OpGraph,
    op_matcher=(lambda g, p: p.tag == "placeholder" or g.tag == p.tag),
    edge_matcher=(lambda g, p: g == p),
) -> Iterator[Dict[OpId, OpId]]:
    return rx.vf2_mapping(
        self._rx_graph,
        pattern._rx_graph,
        subgraph=True,
        node_matcher=op_matcher,
        edge_matcher=edge_matcher,
    )


class RewriteRule:
    def __init__(self) -> None:
        self.p_subgraph, self.p_inp_ids, self.p_out_ids = self._init_pattern_info()
        self.p_inner_ids = set(self.p_subgraph.all_ids()).difference(
            self.p_inp_ids, self.p_out_ids
        )

    def _init_pattern_info(self) -> Tuple[OpGraph, List[OpId], List[OpId]]:
        raise NotImplementedError()

    def _create_new_subgraph(
        self, inp_ops: List[Operator], pms: MatchResult
    ) -> Tuple[OpGraph, List[OpId], List[OpId]]:
        raise NotImplementedError()

    def _match_op(self, g_op: Operator, p_op: Operator):
        return (
            p_op.tag == "placeholder" or g_op.tag == p_op.tag
        ) and g_op.tag != "fission"

    def _match_edge(self, g_etag: ArgIndex, p_etag: ArgIndex):
        return g_etag == p_etag

    def _check_match_result(self, ms: MatchResult):
        return True

    def match(self, G: OpGraph) -> Iterator[MatchResult]:
        P = self.p_subgraph
        for g2p_maps in match_subgraph(G, P, self._match_op, self._match_edge):
            p2g_maps = {v: k for k, v in g2p_maps.items()}

            def is_convex():
                for p_inn_id in self.p_inner_ids:
                    g_inn_id = p2g_maps[p_inn_id]
                    for g_suc_id in G.suc_ids(g_inn_id):
                        if g_suc_id not in g2p_maps:
                            return False
                return True

            if not is_convex():
                continue

            g_inp_ids = [p2g_maps[i] for i in self.p_inp_ids]
            g_out_ids = [p2g_maps[i] for i in self.p_out_ids]
            ms = MatchResult(G, P, g2p_maps, p2g_maps, g_inp_ids, g_out_ids)
            try:
                ret = self._create_new_subgraph([G[i] for i in g_inp_ids], ms)
            except AssertionError:
                continue
            if not ret:
                continue
            S, s_inp_ids, s_out_ids = ret
            ms.new_subgraph = S
            ms.inp_maps = dict(zip(g_inp_ids, s_inp_ids))
            ms.out_maps = dict(zip(g_out_ids, s_out_ids))
            if not self._check_match_result(ms):
                continue
            yield ms
