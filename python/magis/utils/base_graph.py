from typing import List, Dict, Tuple, Set, Generic, TypeVar, Union
from copy import copy as _shallow_copy

import numpy as np
import rustworkx as rx  # refer to https://www.rustworkx.org/# for APIs

_GT = TypeVar("_GT")
_VT = TypeVar("_VT")
_ET = TypeVar("_ET")

NodeId = int
EdgeId = int
Edge = Tuple[NodeId, NodeId, _ET]


class BaseGraph(Generic[_GT, _VT, _ET]):
    def __init__(self, rx_graph: rx.PyDiGraph = None) -> None:
        super().__init__()
        self.reset(rx_graph)

    def reset(self, rx_graph: rx.PyDiGraph = None):
        if rx_graph is None:
            rx_graph = rx.PyDiGraph()
        self._rx_graph = rx_graph
        self._reset_basic_props()
        self._reset_saved_props()

    def _reset_basic_props(self):
        pass

    def _reset_saved_props(self):
        self._des_table: np.ndarray = None
        self._anc_table: np.ndarray = None
        self._fwd_rpo: List[NodeId] = None
        self._bwd_rpo: List[NodeId] = None
        self._live_tables: List[np.ndarray] = [None] * 4

    def shallow_copy(self) -> _GT:
        return _shallow_copy(self)

    def copy(self) -> _GT:
        ret: BaseGraph = self.shallow_copy()
        ret._rx_graph = self._rx_graph.copy()
        return ret

    def may_copy(self, inplace=False) -> _GT:
        self = self if inplace else self.copy()
        self._reset_saved_props()
        return self

    @property
    def des_table(self):
        if self._des_table is None:
            self._des_table = self._get_des_table()
        return self._des_table

    @property
    def anc_table(self):
        if self._anc_table is None:
            self._anc_table = self._get_des_table(fwd=True)
        return self._anc_table

    def _get_des_table(self, fwd=False):
        f_adj = self.pre_ids if fwd else self.suc_ids
        des = np.zeros([self.max_id() + 1] * 2, dtype="bool")
        for v in self.fwd_rpo_topo_order if fwd else self.bwd_rpo_topo_order:
            des_v = des[v]
            des_v[v] = True
            us = f_adj(v)
            if len(us) <= 0:
                continue
            des_v[:] |= np.logical_or.reduce(des[us])
        return des

    @property
    def fwd_rpo_topo_order(self):
        if self._fwd_rpo is None:
            self._fwd_rpo = self.get_rpo_topo_order(fwd=True)
        return self._fwd_rpo

    @property
    def bwd_rpo_topo_order(self):
        if self._bwd_rpo is None:
            self._bwd_rpo = self.get_rpo_topo_order(fwd=False)
        return self._bwd_rpo

    def get_live_table(self, policy=0, cache=True):
        assert 0 <= policy < 4
        if cache and self._live_tables[policy] is not None:
            return self._live_tables[policy]
        # ~anc, des, anc, ~des
        v_fan = self.anc_table if policy % 2 == 0 else self.des_table
        ws_fan = self.des_table if policy // 2 == 0 else self.anc_table
        if policy == 0 or policy == 3:
            mask = np.zeros([self.max_id() + 1], dtype="bool")
            mask[self.all_ids()] = 1
            v_fan = mask & ~v_fan
        ret = np.zeros([self.max_id() + 1] * 2, dtype="bool")
        for v in self.all_ids():
            ws = self.suc_ids(v)
            if len(ws) > 0:
                ret[v] = (
                    v_fan[v] & ~np.logical_and.reduce(ws_fan[ws])
                    if policy // 2 == 0
                    else np.logical_or.reduce(ws_fan[ws]) & ~v_fan[v]
                )
            ret[v, v] = 1
        if cache:
            self._live_tables[policy] = ret
        return ret

    def add_node(self, nd: _VT) -> NodeId:
        self._reset_saved_props()
        return self._rx_graph.add_node(nd)

    def add_nodes(self, nds: List[_VT]) -> List[NodeId]:
        self._reset_saved_props()
        return self._rx_graph.add_nodes_from(nds)

    def add_edge(self, src_id: NodeId, dst_id: NodeId, etag: _ET) -> EdgeId:
        self._reset_saved_props()
        return self._rx_graph.add_edge(src_id, dst_id, etag)

    def add_edges(self, edges: List[Edge]) -> List[EdgeId]:
        self._reset_saved_props()
        return self._rx_graph.add_edges_from(edges)

    def remove_node(self, nid: NodeId):
        self._reset_saved_props()
        self._rx_graph.remove_node(nid)

    def remove_edge(self, src_id: NodeId, dst_id: NodeId):
        self._reset_saved_props()
        self._rx_graph.remove_edge(src_id, dst_id)

    def remove_edges(self, edges: List[Tuple[NodeId, NodeId]]):
        self._reset_saved_props()
        self._rx_graph.remove_edges_from(edges)

    def get_node(self, nid: NodeId) -> _VT:
        return self[nid]

    def set_node(self, nid: NodeId, nd: _VT, inplace=True) -> _GT:
        self = self if inplace else self.copy()
        self[nid] = nd
        return self

    def __getitem__(self, nid: NodeId) -> _VT:
        return self._rx_graph[nid]

    def __setitem__(self, nid: NodeId, nd: _VT):
        self._reset_saved_props()
        self._rx_graph[nid] = nd

    def pre_edges(self, nid: NodeId) -> List[Tuple[NodeId, _ET]]:
        return [(p, e) for p, _, e in self._rx_graph.in_edges(nid)]

    def etag_between(self, src_id: NodeId, dst_id: NodeId) -> _ET:
        return self._rx_graph.get_edge_data(src_id, dst_id)

    def inp_degree(self, nid: NodeId) -> int:
        return self._rx_graph.in_degree(nid)

    def out_degree(self, nid: NodeId) -> int:
        return self._rx_graph.out_degree(nid)

    @property
    def num_nodes(self) -> int:
        return self._rx_graph.num_nodes()

    def all_nodes(self) -> List[_VT]:
        return self._rx_graph.nodes()

    # get sub-graph and map from original graph node-ids to sub-graph node-ids
    def subgraph(
        self, nids: List[NodeId], return_id_map=False
    ) -> Union[_GT, Tuple[_GT, Dict[NodeId, NodeId]]]:
        G: BaseGraph = self.shallow_copy()
        G.reset(self._rx_graph.subgraph(list(nids)))
        if return_id_map:
            return G, dict((_id, i) for i, _id in enumerate(sorted(set(nids))))
        return G

    def pre_ids(self, nid: NodeId) -> List[NodeId]:
        return self._rx_graph.predecessor_indices(nid)

    def suc_ids(self, nid: NodeId) -> List[NodeId]:
        return self._rx_graph.successor_indices(nid)

    def anc_ids(self, nid: NodeId) -> Set[NodeId]:
        return rx.ancestors(self._rx_graph, nid)

    def des_ids(self, nid: NodeId) -> Set[NodeId]:
        return rx.descendants(self._rx_graph, nid)

    def all_ids(self) -> List[NodeId]:
        return self._rx_graph.node_indices()

    def max_id(self) -> NodeId:
        return self._rx_graph.node_indices()[-1]

    def all_inp_ids(self) -> List[NodeId]:
        return [i for i in self.all_ids() if self.inp_degree(i) == 0]

    def all_out_ids(self) -> List[NodeId]:
        return [i for i in self.all_ids() if self.out_degree(i) == 0]

    def get_rpo_topo_order(self, fwd=True, init_ids=None) -> List[NodeId]:
        vis = set()
        ret = []
        f_next = self.suc_ids if fwd else self.pre_ids
        if init_ids is None:
            init_ids = self.all_inp_ids() if fwd else self.all_out_ids()
        if not isinstance(init_ids, (list, tuple)):
            init_ids = [init_ids]

        def po_dfs(cur):
            if cur in vis:
                return
            vis.add(cur)
            for nxt in f_next(cur):
                po_dfs(nxt)
            ret.append(cur)

        for inp in init_ids:
            po_dfs(inp)
        ret.reverse()
        return ret

    def connected_components(self) -> List[Set[NodeId]]:
        return rx.weakly_connected_components(self._rx_graph)

    def is_connected(self):
        return rx.is_weakly_connected(self._rx_graph)
