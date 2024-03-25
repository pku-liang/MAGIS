from typing import Dict, List, Tuple, Union, Set, Iterator
from collections import defaultdict
from dataclasses import dataclass

import numpy as np

from ..operators import PlaceholderOp, FissionOp, FissionOutOp, StoreOp, LoadOp
from ..op_graph import OpGraph, OpId
from ..dim_graph import DimGraph
from ..utils import LOG
from .. import utils


def _bwd_infer_fission_dims(G: OpGraph, out_ids, out_fs_dims):
    fs_dims = defaultdict(set, {i: set(ds) for i, ds in zip(out_ids, out_fs_dims)})
    for i in G.bwd_rpo_topo_order:
        ds = fs_dims[i]
        pre_ds = G[i].infer_input_linked_dims(ds)
        for ids, ii in zip(pre_ds, G.arg_ids(i)):
            fs_dims[ii].update(ids)
    return fs_dims


def _fwd_infer_fission_dims(G: OpGraph, inp_ids, inp_fs_dims):
    fs_dims = defaultdict(set, {i: set(ds) for i, ds in zip(inp_ids, inp_fs_dims)})
    for i in G.fwd_rpo_topo_order:
        pre_ds = [fs_dims[pid] for pid in G.arg_ids(i)]
        ds = G[i].infer_output_linked_dims(pre_ds)
        fs_dims[i].update(ds)
    return fs_dims


def infer_fission_dims(
    G: OpGraph,
    inp_ids=(),
    out_ids=(),
    inp_fs_dims=None,
    out_fs_dims=None,
    all_fs_dims=None,
):
    assert any([inp_fs_dims, out_fs_dims, all_fs_dims])
    if all_fs_dims:
        new_inp_fs_dims = [all_fs_dims[i] for i in inp_ids]
        new_out_fs_dims = [all_fs_dims[i] for i in out_ids]
        if inp_fs_dims:
            assert new_inp_fs_dims == inp_fs_dims
        if out_fs_dims:
            assert new_out_fs_dims == out_fs_dims
        inp_fs_dims = new_inp_fs_dims
        out_fs_dims = new_out_fs_dims
    elif out_fs_dims and out_ids:
        all_fs_dims = _bwd_infer_fission_dims(G, out_ids, out_fs_dims)
        new_inp_fs_dims = [all_fs_dims[i] for i in inp_ids]
        if inp_fs_dims:
            assert new_inp_fs_dims == inp_fs_dims, (new_inp_fs_dims, inp_fs_dims)
        inp_fs_dims = new_inp_fs_dims
    else:
        all_fs_dims = _fwd_infer_fission_dims(G, inp_ids, inp_fs_dims)
        out_fs_dims = [all_fs_dims[i] for i in out_ids]
    return inp_fs_dims, out_fs_dims, all_fs_dims


def _rewrite_graph_op_shapes(
    G: OpGraph, inp_ids, out_ids, fs_dims, ori_len, new_len, inplace=False
):
    G = G.may_copy(inplace=inplace)
    for v in G.fwd_rpo_topo_order:
        old_op = G[v]
        shape = list(old_op.out_shape)
        dims = fs_dims[v]
        if not old_op.is_fission():
            for d in dims:
                if d < 0:
                    continue
                assert shape[d] == ori_len, (v, d, shape[d], ori_len, new_len)
                shape[d] = new_len
        if v in inp_ids:
            assert not old_op.is_fission()
            new_op = PlaceholderOp(shape, is_weight=old_op.is_weight())
            if len(dims) == 0:
                new_op._out_memory = 0  # no slicing, directly reuse input op
            # else:
            #     new_op._out_memory = 0
        elif isinstance(old_op, FissionOp):
            assert len(dims) == 1
            if new_len < old_op.fs_factor:
                new_op = update_fs_factor(old_op, new_len, inplace=False)
            else:
                new_op = old_op.may_copy(inplace=False)
            new_op.fs_length = new_len
            new_real_out_shapes = []
            for shape, dims in zip(new_op.real_out_shapes, new_op.out_fs_dims):
                shape = list(shape)
                for d in dims:
                    if d < 0:
                        continue
                    assert shape[d] == ori_len
                    shape[d] = new_len
                new_real_out_shapes.append(tuple(shape))
            new_op.real_out_shapes = new_real_out_shapes
        else:
            new_op = old_op.may_copy(inplace=False)
            new_op.out_shape = tuple(shape)
            new_op.inp_shapes = [G[u].out_shape for u in G.arg_ids(v)]
            if v in out_ids:
                assert not old_op.is_fission()
                assert len(dims) == 1
            # else:
            #     assert all(d >= 0 for d in dims), (v, G[v].key, dims)
        G[v] = new_op
    return G


def update_fs_factor(fs_op: FissionOp, new_factor, inplace=False):
    assert fs_op.fs_length % new_factor == 0
    fs_op = fs_op.may_copy(inplace=inplace)
    all_fs_dims = infer_fission_dims(
        fs_op.fs_graph,
        fs_op.fs_inp_ids,
        fs_op.fs_out_ids,
        fs_op.inp_fs_dims,
        fs_op.out_fs_dims,
    )[-1]
    fs_op.fs_graph = _rewrite_graph_op_shapes(
        fs_op.fs_graph,
        fs_op.fs_inp_ids,
        fs_op.fs_out_ids,
        all_fs_dims,
        fs_op.fs_factor,
        new_factor,
        inplace=False,
    )
    fs_op.fs_factor = new_factor
    return fs_op


def apply_fission(
    G: OpGraph = None,
    G_fs_all_ids: Set[OpId] = None,
    G_fs_inp_ids: List[OpId] = None,
    G_fs_out_ids: List[OpId] = None,
    all_fs_dims: Dict[OpId, Set[int]] = None,
    inp_fs_dims: List[Set[int]] = None,
    out_fs_dims: List[Set[int]] = None,
    fs_length: int = None,
    fs_factor: int = None,
    fs_nparts: int = None,
    inplace=False,
    return_fs_op_id=False,
    dim_graph: DimGraph = None,
    dom_id: OpId = None,
    check_legality=True,
) -> Union[OpGraph, Tuple[OpGraph, OpId]]:
    # extract arguments from DimGraph
    if dim_graph is not None:
        G = dim_graph.op_graph
        dim_map = dim_graph.dim_map()
        if dom_id is not None:
            dom_table = dim_graph.op_dom_table
            V = set(dom_table[:, dom_id].nonzero()[0])
            O = set(
                v
                for v in V
                if G.out_degree(v) <= 0 or any(w not in V for w in G.suc_ids(v))
            )
            V.difference_update(*(set(dom_table[:, v].nonzero()[0]) - {v} for v in O))
            O.intersection_update(V)
            I = (
                set(u for v in V - {dom_id} for u in G.pre_ids(v))
                .difference(V)
                .union({dom_id})
            )
        else:
            V = dim_graph.all_op_ids()
            I = {u for v in V for u in G.pre_ids(v)} - V
            O = {v for v in V if len(set(G.suc_ids(v)) - V) > 0} | V.intersection(
                G.all_out_ids()
            )
        G_fs_all_ids, G_fs_inp_ids, G_fs_out_ids = V | I, list(I), list(O)
        G_fs_inp_ids.sort(key=lambda v: G[v].key)
        G_fs_out_ids.sort(key=lambda v: G[v].key)
        out_fs_dims = [{dim_map[v]} for v in G_fs_out_ids]
        inp_fs_dims = all_fs_dims = None
        check_legality = False  # guarantee legality

    # check legality
    def _check_legality():
        # assert all(not G[i].is_fission() for i in G_fs_all_ids)
        assert len(set(G_fs_inp_ids).intersection(G_fs_out_ids)) == 0
        inn_ids = G_fs_all_ids.difference(G_fs_inp_ids, G_fs_out_ids)
        # inner op should be defined and used internally
        for v in inn_ids:
            U = G.pre_ids(v)
            assert len(U) > 0
            for u in U:
                assert u in G_fs_all_ids
            for w in G.suc_ids(v):
                assert w in G_fs_all_ids
        # input op should be defined externally
        for v in G_fs_inp_ids:
            for u in G.pre_ids(v):
                assert u not in G_fs_all_ids
        # output op should be defined internally
        for v in G_fs_out_ids:
            U = G.pre_ids(v)
            assert len(U) > 0
            for u in U:
                assert u in G_fs_all_ids

    if check_legality:
        _check_legality()

    # extract fission-graph and re-map some IDs
    F, G2F_id_map = G.subgraph(G_fs_all_ids, return_id_map=True)
    F_inp_ids = [G2F_id_map[i] for i in G_fs_inp_ids]
    F_out_ids = [G2F_id_map[i] for i in G_fs_out_ids]
    # remove inter-input edges
    F.remove_edges([(u, v) for u in F_inp_ids for v in F.suc_ids(u) if v in F_inp_ids])
    if all_fs_dims is not None:
        all_fs_dims = {v: all_fs_dims[k] for k, v in G2F_id_map.items()}
    inp_fs_dims, out_fs_dims, all_fs_dims = infer_fission_dims(
        F, F_inp_ids, F_out_ids, inp_fs_dims, out_fs_dims, all_fs_dims
    )

    # check length and factor
    if fs_length is None:
        for oid, dims in all_fs_dims.items():
            sp_dim = next((d for d in dims if d >= 0), None)
            if sp_dim is not None:
                fs_length = F[oid].out_shape[sp_dim]
                break
    assert fs_length is not None
    if fs_factor is None:
        assert fs_nparts is not None and fs_length % fs_nparts == 0
        fs_factor = fs_length // fs_nparts
    assert fs_length % fs_factor == 0

    # rewrite op-shapes of fission-graph
    F = _rewrite_graph_op_shapes(
        F, F_inp_ids, F_out_ids, all_fs_dims, fs_length, fs_factor, inplace=True
    )

    # prepare new sub-graph with fission-op
    fs_op = FissionOp(
        *(G, G_fs_inp_ids, G_fs_out_ids),
        *(F, F_inp_ids, F_out_ids),
        *(inp_fs_dims, out_fs_dims),
        *(fs_length, fs_factor),
    )
    S = OpGraph()
    S_inp_ids = S.add_nodes([G[v] for v in G_fs_inp_ids])
    S_fs_op_id = S.add_op_with_args(fs_op, S_inp_ids)
    S_out_ids = [
        S.add_op_with_args(FissionOutOp(fs_op, i, dims), [S_fs_op_id])
        for i, dims in enumerate(out_fs_dims)
    ]

    # replace original sub-graph with new sub-graph with fission-op
    G, S2G_id_map = G.replace_subgraph(
        *(G_fs_all_ids, S),
        dict(zip(G_fs_inp_ids, S_inp_ids)),
        dict(zip(G_fs_out_ids, S_out_ids)),
        return_id_map=True,
        inplace=inplace,
    )
    G_fs_op_id = S2G_id_map[S_fs_op_id]

    if return_fs_op_id:
        return G, G_fs_op_id
    return G


def remove_fission(
    G: OpGraph, G_fs_op_id: OpId, inplace=False, return_subgraph_ids=False
) -> Union[OpGraph, Tuple[OpGraph, List[OpId]]]:
    fs_op: FissionOp = G[G_fs_op_id]
    assert fs_op.is_fission()
    F = fs_op.fs_graph
    F_inp_ids = fs_op.fs_inp_ids
    F_out_ids = fs_op.fs_out_ids

    all_fs_dims = infer_fission_dims(
        *(F, F_inp_ids, F_out_ids),
        fs_op.inp_fs_dims,
        fs_op.out_fs_dims,
    )[-1]

    F = _rewrite_graph_op_shapes(
        *(F, F_inp_ids, F_out_ids),
        all_fs_dims,
        fs_op.fs_factor,
        fs_op.fs_length,
        inplace=False,
    )

    G_fs_inp_ids = G.arg_ids(G_fs_op_id)
    G_fs_out_ids = sorted(G.suc_ids(G_fs_op_id), key=lambda i: G[i].attrs["index"])
    G_fs_all_ids = {*G_fs_inp_ids, G_fs_op_id, *G_fs_out_ids}

    for fi, gi in zip(F_inp_ids, G_fs_inp_ids):
        F[fi] = G[gi]

    G, F2G_id_map = G.replace_subgraph(
        *(G_fs_all_ids, F),
        dict(zip(G_fs_inp_ids, F_inp_ids)),
        dict(zip(G_fs_out_ids, F_out_ids)),
        return_id_map=True,
        inplace=inplace,
    )

    if return_subgraph_ids:
        return G, [F2G_id_map[i] for i in F.all_ids()]
    return G


def remove_deepest_fission(G: OpGraph, inplace=False):
    height = G.fs_height()
    if height <= 0:
        return G
    if height == 1:
        G = G.may_copy(inplace=inplace)
        for fs_id in G.sub_fs_ids:
            G = remove_fission(G, fs_id, inplace=True)
        return G

    def _dfs(G: OpGraph, lv=()):
        if len(lv) == height - 1:
            if len(G.sub_fs_ids) > 0:
                G = G.may_copy(inplace=False)
                for fs_id in G.sub_fs_ids:
                    G = remove_fission(G, fs_id, inplace=True)
                return G, True
            return G, False

        sub_fs_graphs = []
        sub_fs_changes = []
        for fs_id in G.sub_fs_ids:
            fs_op: FissionOp = G[fs_id]
            new_fs_graph, new_change = _dfs(fs_op.fs_graph, (*lv, fs_id))
            sub_fs_graphs.append(new_fs_graph)
            sub_fs_changes.append(new_change)

        changed = any(sub_fs_changes)
        if changed:
            G = G.may_copy(inplace=inplace if len(lv) == 0 else False)
            for fs_id, fs_graph, fs_change in zip(
                G.sub_fs_ids, sub_fs_graphs, sub_fs_changes
            ):
                if fs_change:
                    new_fs_op: FissionOp = G[fs_id].may_copy(inplace=False)
                    new_fs_op.fs_graph = fs_graph
                    G[fs_id] = new_fs_op
        return G, changed

    return _dfs(G)[0]


def apply_swap(self: OpGraph, v: OpId, ws: OpId, inplace=False):
    if not isinstance(ws, list):
        ws = [ws]
    v, ws = utils.to_tuple(v), [utils.to_tuple(w) for w in ws]
    fs_ids = v[:-1]
    assert all(fs_ids == w[:-1] for w in ws)
    v, ws = v[-1], [w[-1] for w in ws]
    G = self.fs_get_graph(fs_ids)
    G = G.may_copy(inplace=False)

    assert not G[v].is_store()
    assert all(not G[w].is_mem_op() for w in ws)

    etags = [G.etag_between(v, w) for w in ws]
    [G.remove_edge(v, w) for w in ws]
    if G.fs_get_op(v).is_load():
        us = G.fs_pre_ids(v)
        assert len(us) == 1
        u = us[0]
        assert G[u].is_store()
        ld = G.add_node(LoadOp(G[u]))
        G.add_edges([(u, ld, 0)] + [(ld, w, e) for w, e in zip(ws, etags)])
    else:
        w_st = None
        for w_ in G.suc_ids(v):
            if G[w_].is_store():
                assert w_st is None
                w_st = w_
        if w_st is not None:
            ld = G.add_node(LoadOp(G[w_st]))
            G.add_edges([(w_st, ld, 0)] + [(ld, w, e) for w, e in zip(ws, etags)])
        else:
            st = G.add_node(StoreOp(G[v]))
            ld = G.add_node(LoadOp(G[st]))
            G.add_edges(
                [(v, st, 0), (st, ld, 0)] + [(ld, w, e) for w, e in zip(ws, etags)]
            )
    return self.fs_set_graph(fs_ids, G, inplace)


def apply_remat(self: OpGraph, v: OpId, ws: OpId, inplace=False, seperate=False):
    if not isinstance(ws, list):
        ws = [ws]
    v, ws = utils.to_tuple(v), [utils.to_tuple(w) for w in ws]
    fs_ids = v[:-1]
    assert all(fs_ids == w[:-1] for w in ws)
    v, ws = v[-1], [w[-1] for w in ws]
    G = self.fs_get_graph(fs_ids)
    G = G.may_copy(inplace=False)

    assert not G[v].is_store()
    assert all(not G[w].is_mem_op() for w in ws)

    etags = [G.etag_between(v, w) for w in ws]
    [G.remove_edge(v, w) for w in ws]
    if not seperate:
        v_ = G.add_op_with_args(G[v].copy(), G.arg_ids(v))
        G.add_edges([(v_, w, e) for w, e in zip(ws, etags)])
        G[v_]._is_remat = True
    else:
        for w, e in zip(ws, etags):
            v_ = G.add_op_with_args(G[v].copy(), G.arg_ids(v))
            G.add_edges([(v_, w, e)])
            G[v_]._is_remat = True

    return self.fs_set_graph(fs_ids, G, inplace)
