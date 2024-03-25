from typing import List, Tuple, Dict, Union
from collections import namedtuple, defaultdict

import numpy as np
import rustworkx as rx

from .utils.base_graph import BaseGraph, NodeId
from .op_graph import OpGraph, OpId, ArgIndex
from . import utils

DimIndex = int
Dimension = namedtuple("Dimension", ["op_id", "d_idx"])
DimId = NodeId


class DimGraph(BaseGraph["DimGraph", Dimension, ArgIndex]):
    def __init__(self, op_graph: OpGraph = None, rx_graph: rx.PyDiGraph = None) -> None:
        super().__init__(rx_graph)
        self.op_graph = None
        if op_graph is not None:
            self.bind_op_graph(op_graph)

    def _reset_saved_props(self):
        super()._reset_saved_props()
        self._op_dom_table: np.ndarray = None

    @property
    def op_dom_table(self):
        if self._op_dom_table is None:
            G = self.op_graph
            dom_table = self._op_dom_table = G.dom_table.copy()
            op_ids = self.all_op_ids()
            for v in G.all_ids():
                if G[v].is_placeholder():
                    continue
                if v not in op_ids:
                    dom_table[v, :] = 0
                    dom_table[:, v] = 0
        return self._op_dom_table

    def bind_op_graph(self, G: OpGraph):
        if G is None:
            return
        self.reset()
        self.op_graph = G

        grps = utils.UnionFindSet[Dimension]()
        dim_links: Dict[Dimension, List[Tuple[Dimension, ArgIndex]]] = defaultdict(list)
        dim2id = dict()

        des = G.des_table

        for v in G.bwd_rpo_topo_order:
            for v_cur_d_idx in G[v].loop_dims:
                v_cur_dim = Dimension(v, v_cur_d_idx)
                v_grp_ids = []
                suc_grps = []
                success = True
                for w_dim, _ in dim_links[v_cur_dim]:
                    w_grp = grps.find_root(w_dim)
                    if w_grp in suc_grps:
                        continue
                    w_grp_ids = grps[w_grp]
                    if (
                        des[np.ix_(v_grp_ids, w_grp_ids)].any()
                        or des[np.ix_(w_grp_ids, v_grp_ids)].any()
                    ):
                        success = False
                        break
                    v_grp_ids.extend(w_grp_ids)
                    suc_grps.append(w_grp)
                if success:
                    v_grp_ids.append(v)
                    grps[v_cur_dim] = v_grp_ids
                    for grp in suc_grps:
                        grps.union_trees(grp, v_cur_dim)
                    dim2id[v_cur_dim] = v_cur_did = self.add_node(v_cur_dim)
                    for w_dim, arg_idx in dim_links[v_cur_dim]:
                        w_did = dim2id[w_dim]
                        self.add_edge(v_cur_did, w_did, arg_idx)
                else:
                    grps[v_cur_dim] = [v]
                    dim2id[v_cur_dim] = self.add_node(v_cur_dim)

            for (i, u), links in zip(enumerate(G.arg_ids(v)), G[v]._dim_links):
                for u_d, v_cur_d_idx in links:
                    dim_links[(u, u_d)].append(((v, v_cur_d_idx), i))

    def get_dim_op(self, did: Union[DimId, Dimension]):
        dim = did if isinstance(did, Dimension) else self[did]
        return self.op_graph[dim.op_id]

    def get_dim_len(self, did: Union[DimId, Dimension]):
        dim = did if isinstance(did, Dimension) else self[did]
        return self.op_graph[dim.op_id].loop_lens[dim.d_idx]

    def get_connected_dim_len(self):
        assert self.is_connected()
        dim_lens = [self.get_dim_len(did) for did in self.all_ids()]
        ret = next(d for d in dim_lens if d > 0)
        assert all(d == ret for d in dim_lens if d > 0), dim_lens
        return ret

    def all_op_ids(self):
        return {dim.op_id for dim in self.all_nodes()}

    def dim_map(self) -> Dict[OpId, DimIndex]:
        return {dim.op_id: dim.d_idx for dim in self.all_nodes()}

    def graphviz(self, name, fmt="png", render=True):
        dot_str = self._rx_graph.to_dot(
            node_attr=lambda d: dict(
                fontname="consolas", label=str(tuple(d)), shape="box"
            )
        )
        return utils.render_dot(dot_str, name=name, fmt=fmt, render=render)
