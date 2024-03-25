import math
from functools import reduce
from collections import defaultdict
from typing import List, Dict, Tuple, Optional, Union

import rustworkx as rx  # refer to https://qiskit.org/ecosystem/rustworkx/ for APIs
import numpy as np

from .operators import *
from . import utils
from .utils.base_graph import BaseGraph, NodeId

OpId = NodeId
ArgIndex = Optional[int]
# hierarchy sched, key: fiss_ids, value: sched of fiss_graph
HierSched = Dict[List[OpId], List[OpId]]
# sched of the whole graph
FlatSched = List[Union[OpId, List[OpId]]]


class OpGraph(BaseGraph["OpGraph", Operator, ArgIndex]):
    def __init__(self, rx_graph: rx.PyDiGraph = None) -> None:
        super().__init__(rx_graph)

    # ----------------------------------------------------------------
    # [BEGIN] properties

    def _reset_basic_props(self):
        self._fs_inp_ids = []  # input ids of fission graph
        self._fs_out_ids = []  # output ids of fission graph

    def _reset_saved_props(self):
        super()._reset_saved_props()
        self._sched: List[OpId] = None
        self._digest: bytes = None
        self._entry_id: OpId = None
        self._dom_table: np.ndarray = None
        self._post_dom_table: np.ndarray = None
        self._imm_dom_map: Dict[OpId, OpId] = None
        self._imm_post_dom_map: Dict[OpId, OpId] = None
        self._dim_graph = None
        self._sub_fs_ids: List[OpId] = None  # fission op ids of this graph

    def copy(self):
        ret = super().copy()
        ret._fs_inp_ids = self._fs_inp_ids.copy()
        ret._fs_out_ids = self._fs_out_ids.copy()
        return ret

    @property
    def sched(self):
        if self._sched is None:
            self._sched = self.fwd_rpo_topo_order
        return self._sched

    @property
    def digest(self):
        if self._digest is None:
            self._digest = self._get_digest()
        return self._digest

    @property
    def entry_id(self):
        if self._entry_id is None:
            self._entry_id = self._get_entry_id()
        return self._entry_id

    @property
    def sub_fs_ids(self):
        if self._sub_fs_ids is None:
            self._sub_fs_ids = self._get_sub_fs_ids()
        return self._sub_fs_ids

    @property
    def sub_fs_ops(self) -> List[Tuple[OpId, FissionOp]]:
        return [(fs_id, self[fs_id]) for fs_id in self.sub_fs_ids]

    def _get_sub_fs_ids(self):
        return [v for v in self.all_ids() if self[v].is_fission()]

    def _get_digest(self):
        op_digests = dict()
        for op_id in self.fwd_rpo_topo_order:
            op_digests[op_id] = self[op_id]._merge_args_digest(
                # [self[a].digest for a in self.arg_ids(op_id)]
                [op_digests[a] for a in self.arg_ids(op_id)]
            )
        return utils.digest(b"".join(sorted(op_digests.values())))

    def _get_entry_id(self):
        return min(v for v in self.all_ids() if not self[v].is_weight())

    @property
    def dom_table(self):
        if self._dom_table is None:
            self._dom_table = self._get_dom_table(fwd=True)
        return self._dom_table

    @property
    def post_dom_table(self):
        if self._post_dom_table is None:
            self._post_dom_table = self._get_dom_table(fwd=False)
        return self._post_dom_table

    def _get_dom_table(self, fwd=True):
        # ignore input nodes if more than 1
        ignored_ids = set(self.all_inp_ids()).difference([self.entry_id])
        dom = np.zeros([self.max_id() + 1] * 2, dtype="bool")
        f_adj = self.pre_ids if fwd else self.suc_ids
        for v in self.fwd_rpo_topo_order if fwd else self.bwd_rpo_topo_order:
            if v in ignored_ids:
                continue
            dom_v = dom[v]
            us = [u for u in f_adj(v) if u not in ignored_ids]
            if len(us) > 0:
                dom_v[:] = np.logical_and.reduce(dom[us])
            dom_v[v] = 1
        return dom

    @property
    def imm_dom_map(self):
        if self._imm_dom_map is None:
            self._imm_dom_map = self._get_imm_dom_map(fwd=True)
        return self._imm_dom_map

    @property
    def imm_post_dom_map(self):
        if self._imm_post_dom_map is None:
            self._imm_post_dom_map = self._get_imm_dom_map(fwd=False)
        return self._imm_post_dom_map

    def _get_imm_dom_map(self, fwd=True):
        ignored_ids = set(self.all_inp_ids()).difference([self.entry_id])
        topo_ids = np.full([self.max_id() + 1], -1, dtype="int32")
        idom_map: Dict[OpId, OpId] = dict()
        topo_order = self.fwd_rpo_topo_order if fwd else self.bwd_rpo_topo_order
        f_adj = self.pre_ids if fwd else self.suc_ids
        for i, v in enumerate(topo_order):
            topo_ids[v] = i

        def intersect(v1, v2):
            if v1 is None or v2 is None:
                return None
            while v1 != v2:
                i1 = topo_ids[v1]
                while i1 < topo_ids[v2]:
                    v2 = idom_map.get(v2, None)
                    if v2 is None:
                        return None
                i2 = topo_ids[v2]
                while i2 < topo_ids[v1]:
                    v1 = idom_map.get(v1, None)
                    if v1 is None:
                        return None
            return v1

        for v in topo_order:
            if v in ignored_ids:
                continue
            us = [u for u in f_adj(v) if u not in ignored_ids]
            if len(us) <= 0:
                continue
            new_idom = us[0]
            for u in us[1:]:
                new_idom = intersect(new_idom, u)
            if new_idom is not None:
                idom_map[v] = new_idom

        return idom_map

    @property
    def dim_graph(self):
        if self._dim_graph is None:
            from .dim_graph import DimGraph

            self._dim_graph = DimGraph(self)
        return self._dim_graph

    def get_memory_heats(self, policy=0):
        lives = self.get_live_table(policy)
        sizes = np.zeros([self.max_id() + 1], dtype="float32")
        for v in self.all_ids():
            op = self[v]
            if isinstance(op, FissionOp):
                sizes[v] = op.fs_graph.get_memory_heats(policy).sum()
            else:
                sizes[v] = op.out_memory
        return lives[:, (lives.T @ sizes).argmax()] * sizes

    # [END] properties
    # ----------------------------------------------------------------

    # ----------------------------------------------------------------
    # [BEGIN] basic graph operations

    def arg_ids(self, op_id: OpId) -> List[OpId]:
        return [oi for oi, _ in sorted(self.pre_edges(op_id), key=lambda t: t[1])]

    def all_wgt_ids(self) -> List[OpId]:
        return [oi for oi in self.all_ids() if self[oi].is_weight()]

    def add_op_with_args(
        self, op: Operator, arg_ids: List[OpId], arg_indices: List[ArgIndex] = None
    ) -> OpId:
        if arg_indices is None:
            arg_indices = list(range(len(arg_ids)))
        assert len(arg_indices) == len(arg_ids)
        op_id = self.add_node(op)
        self.add_edges([(p, op_id, a) for p, a in zip(arg_ids, arg_indices)])
        return op_id

    # [END] basic graph operations
    # ----------------------------------------------------------------

    # ----------------------------------------------------------------
    # [BEGIN] operations about fission graph

    def fs_get_graph(self, fs_ids):
        for fs_id in fs_ids:
            fs_op: FissionOp = self[fs_id]
            assert fs_op.is_fission()
            self = fs_op.fs_graph
        return self

    def fs_get_op(self, ids):
        ids = utils.to_tuple(ids)
        G = self.fs_get_graph(ids[:-1])
        return G[ids[-1]]

    def fs_set_graph(self, fs_ids, G, inplace=False) -> "OpGraph":
        fs_ids = utils.to_tuple(fs_ids)
        if len(fs_ids) <= 0:
            return G
        fs_id = fs_ids[0]
        fs_op: FissionOp = self[fs_id]
        assert fs_op.is_fission()
        fs_op = fs_op.may_copy(inplace=False)
        fs_op.fs_graph = fs_op.fs_graph.fs_set_graph(fs_ids[1:], G, inplace=False)
        return self.set_node(fs_id, fs_op, inplace=inplace)

    def fs_set_op(self, ids, op: Operator, inplace=False) -> "OpGraph":
        ids = utils.to_tuple(ids)
        fs_id = ids[0]
        if len(ids) > 1:
            fs_op: FissionOp = self[fs_id]
            assert fs_op.is_fission()
            fs_op = fs_op.may_copy(inplace=False)
            fs_op.fs_graph = fs_op.fs_graph.fs_set_op(ids[1:], op, inplace=False)
            op = fs_op
        return self.set_node(fs_id, op, inplace=inplace)

    def fs_suc_ids(self, ids):
        if not isinstance(ids, (list, tuple)):
            return self.suc_ids(ids)
        level = ids[:-1]
        G = self.fs_get_graph(level)
        return [(*level, sid) for sid in G.suc_ids(ids[-1])]

    def fs_pre_ids(self, ids):
        if not isinstance(ids, (list, tuple)):
            return self.pre_ids(ids)
        level = ids[:-1]
        G = self.fs_get_graph(level)
        return [(*level, sid) for sid in G.pre_ids(ids[-1])]

    def fs_height(self):
        return max(len(level) for _, level, _ in self.iter_fs_graphs())

    def iter_fs_graphs(
        self, level: List[int] = (), last_fs_op: FissionOp = None, order="pre"
    ):
        if "pre" in order:
            yield self, level, last_fs_op
        for fs_id in self.sub_fs_ids:
            fs_op: FissionOp = self[fs_id]
            assert fs_op.is_fission()
            G = fs_op.fs_graph
            yield from G.iter_fs_graphs((*level, fs_id), fs_op, order)
        if "post" in order:
            yield self, level, last_fs_op

    def import_hier_sched(self, hier_sched: HierSched):
        for fs_ids, sched in hier_sched.items():
            self.fs_get_graph(fs_ids)._sched = tuple(sched)
        return self

    def import_flat_sched(self, flat_shed: FlatSched):
        hier_sched = defaultdict(list)
        for ids in flat_shed:
            ids = utils.to_tuple(ids)
            hier_sched[ids[:-1]].append(ids[-1])
        self.import_hier_sched(hier_sched)
        return self

    def export_hier_sched(self) -> HierSched:
        hier_sched = dict()
        for G, level, _ in self.iter_fs_graphs():
            hier_sched[level] = tuple(G.sched)
        return hier_sched

    def export_flat_sched(self, level=(), unpack_unary=True, ret=None) -> FlatSched:
        if ret is None:
            ret = []
        for oid in self.sched:
            ret.append(oid if (len(level) == 0 and unpack_unary) else (*level, oid))
            op = self[oid]
            if isinstance(op, FissionOp):
                op.fs_graph.export_flat_sched((*level, oid), unpack_unary, ret)
        return ret

    # [END] operations about fission graph
    # ----------------------------------------------------------------

    # ----------------------------------------------------------------
    # [BEGIN] transformation

    def _maintain_fs_infos(self, f_map):
        for book in (self._fs_inp_ids, self._fs_out_ids):
            for i in range(len(book)):
                new_value = f_map(book[i])
                if new_value is not None:
                    book[i] = new_value

    # inplaced replace subgraph
    def replace_subgraph(
        self,
        subgraph_op_ids: Set[OpId],
        new_subgraph: "OpGraph",
        inp_maps: Dict[OpId, OpId],  # from original graph to new_subgraph
        out_maps: Dict[OpId, OpId],  # from original graph to new_subgraph
        return_id_map: bool = False,
        inplace=False,
    ) -> Union["OpGraph", Tuple["OpGraph", Dict[OpId, OpId]]]:
        self = self.may_copy(inplace=inplace)
        # collect in-bound and out-bound edges
        ori_gid_to_new_subgid = {**inp_maps, **out_maps}
        ori_id_to_edges = [
            (i1, self._rx_graph.in_edges(i0), self._rx_graph.out_edges(i0))
            for i0, i1 in ori_gid_to_new_subgid.items()
        ]
        # remove original subgraph
        self._rx_graph.remove_nodes_from(list(subgraph_op_ids))
        # create a new graph and get mappings from new_subgraph to new graph
        new_subgid_to_new_gid: Dict[OpId, OpId] = self._rx_graph.compose(
            new_subgraph._rx_graph, dict()
        )
        # add edges
        for i1, ies, oes in ori_id_to_edges:
            for src, _, tag in ies:
                if src not in subgraph_op_ids:
                    self._rx_graph.add_edge(src, new_subgid_to_new_gid[i1], tag)
            for _, dst, tag in oes:
                if dst not in subgraph_op_ids:
                    self._rx_graph.add_edge(new_subgid_to_new_gid[i1], dst, tag)
        # tracing some important ids
        for book in (self._fs_inp_ids, self._fs_out_ids):
            for i in range(len(book)):
                i1 = ori_gid_to_new_subgid.get(book[i], None)
                if i1 is not None:
                    book[i] = new_subgid_to_new_gid[i1]
        if not return_id_map:
            return self
        return self, new_subgid_to_new_gid

    def backward(self, out_id, inplace=False, update_weight=True) -> "OpGraph":
        G = self.may_copy(inplace=inplace)
        need_grad = set()
        for op_id in G.fwd_rpo_topo_order:
            if G[op_id].is_weight():
                need_grad.add(op_id)
            elif any(pre_id in need_grad for pre_id in G.pre_ids(op_id)):
                need_grad.add(op_id)

        bwd_rpo = G.get_rpo_topo_order(fwd=False, init_ids=out_id)
        bwd_rpo = [_id for _id in bwd_rpo if _id in need_grad]

        op_out_grads: Dict[OpId, List[OpId]] = defaultdict(list)
        op_out_grads[out_id].append(G.placeholder(G[out_id].out_shape, "grad"))
        for y_id in bwd_rpo:
            y = G[y_id]
            x_ids = G.arg_ids(y_id)
            xs = [G[i] for i in x_ids]
            g_id = reduce(lambda a, b: G.add(a, b), op_out_grads[y_id])
            g = G[g_id]
            op2id = dict((*zip(xs, x_ids), (y, y_id), (g, g_id)))
            if y.multi_out_shapes() is None:
                xgrads, links = y.backward(g, xs)
            else:
                ind_out_ids = sorted(G.suc_ids(y_id), key=lambda i: G[i].attrs["index"])
                ind_outs = [G[i] for i in ind_out_ids]
                op2id.update(zip(ind_outs, ind_out_ids))
                xgrads, links = y.backward(g, xs, ind_outs)
            tmp_need_grad = {xg for xg, i in zip(xgrads, x_ids) if i in need_grad}
            for out, inps in reversed(links):
                if out in tmp_need_grad:
                    tmp_need_grad.update(inps)
            for out, inps in links:
                if out not in tmp_need_grad:
                    continue
                op2id[out] = G.add_op_with_args(out, [op2id[inp] for inp in inps])
            for x_id, xg in zip(x_ids, xgrads):
                if x_id in need_grad:
                    xg_id = op2id[xg]
                    op_out_grads[x_id].append(xg_id)
                    if update_weight and G[x_id].is_weight():
                        G.ewise_binop("add_", x_id, xg_id)  # inplace update

        return G

    # [END] transformation
    # ----------------------------------------------------------------

    # ----------------------------------------------------------------
    # [BEGIN] visiualization

    def dump_op(self, op_id: OpId, indent=0) -> str:
        return self[op_id].dump(op_id, self.arg_ids(op_id), indent)

    def dump(self, indent=0) -> str:
        ret = ""
        for op_id in self.sched:
            ret += self.dump_op(op_id, indent) + "\n"
        return ret

    def graphviz_dom_tree(self, name="dom_tree", fmt="png", render=True, fwd=True):
        from graphviz import Digraph

        idom = self.imm_dom_map if fwd else self.imm_post_dom_map
        dot = Digraph()
        for v in {*idom.keys(), *idom.values()}:
            # op = self[v]
            attrs = dict(name=str(v), label=str(v), shape="box", fontname="consolas")
            attrs.update(getattr(self[v], "dot_attrs", dict()))
            dot.node(**attrs)
        for v, u in idom.items():
            if fwd:
                dot.edge(str(u), str(v))
            else:
                dot.edge(str(v), str(u))
        if render:
            dot.render(filename=name, format=fmt, cleanup=True)
        return dot

    def graphviz(
        self,
        name="graph",
        fmt="png",
        render=True,
        short_op_attrs=False,
        show_fs_graphs=False,
        show_op_attrs=False,
        simple_mode=False,
    ):
        from graphviz import Digraph

        dot = Digraph()

        def attrs2str(attrs: dict):
            attrs = dict(attrs)
            del_ks = []
            for k, v in attrs.items():
                if v is None:
                    del_ks.append(k)
            for k in del_ks:
                attrs.pop(k)
            return (
                f"{{{';'.join((str(k)[0] if short_op_attrs else str(k)) + '=' + str(v) for k, v in attrs.items())}}}"
                if show_op_attrs
                else ""
            )

        def draw(G: OpGraph, level, dot: Digraph):
            _id2name = dict()

            def id2name(_id: OpId):
                if _id not in _id2name:
                    op = G[_id]
                    _name = f"{(*level, _id)}"
                    attrs = dict(
                        name=_name,
                        label=(
                            f"id: {_name}\n{op.tag}\n{list(op.out_shape)}\n{attrs2str(op.attrs)}"
                            if not simple_mode
                            else f"{_name}"
                        ),
                        shape="box",
                        fontname="consolas",
                    )
                    attrs.update(getattr(op, "dot_attrs", dict()))
                    dot.node(**attrs)
                    _id2name[_id] = _name
                return _id2name[_id]

            for cur_id in G.all_ids():
                for pre_id, etag in G.pre_edges(cur_id):
                    dot.edge(
                        id2name(pre_id),
                        id2name(cur_id),
                        label=str(etag) if not simple_mode else None,
                        fontname="consolas",
                    )

        if show_fs_graphs:
            for G, lv, _ in self.iter_fs_graphs(order="pre"):
                with dot.subgraph(
                    name=f"cluster_{lv}",
                    graph_attr=dict(
                        label=f"fission-input-ids: {G._fs_inp_ids}\nfission-output-ids: {G._fs_out_ids}",
                        fontname="consolas",
                    ),
                ) as dot_:
                    draw(G, lv, dot_)
        else:
            draw(self, (), dot)

        if render:
            dot.render(filename=name, format=fmt, cleanup=True)

        return dot

    # [END] visiualization
    # ----------------------------------------------------------------

    # ----------------------------------------------------------------
    # [BEGIN] operator builders

    def placeholder(self, shape, name_hint="p", is_weight=False):
        return self.add_node(PlaceholderOp(shape, name_hint, is_weight=is_weight))

    def conv2d(
        self,
        img_id,
        wgt_id,
        layout=None,
        padding=0,
        stride=1,
        dilation=1,
    ):
        layout = layout or Operator.DEFAULT_IMAGE_LAYOUT
        op = Conv2dOp(
            self[img_id],
            self[wgt_id],
            layout,
            padding,
            stride,
            dilation,
        )
        return self.add_op_with_args(op, [img_id, wgt_id])

    def pool2d(self, tag, img_id, kernel_size, layout=None, padding=0, stride=1):
        layout = layout or Operator.DEFAULT_IMAGE_LAYOUT
        op = Pool2dOp(tag, self[img_id], kernel_size, layout, padding, stride)
        return self.add_op_with_args(op, [img_id])

    def max_pool2d(self, img_id, kernel_size, layout=None, padding=0, stride=1):
        return self.pool2d("max", img_id, kernel_size, layout, padding, stride)

    def avg_pool2d(self, img_id, kernel_size, layout=None, padding=0, stride=1):
        return self.pool2d("avg", img_id, kernel_size, layout, padding, stride)

    def matmul(self, a_id, b_id, trans_a=False, trans_b=False, trans_c=False):
        op = MatmulOp(self[a_id], self[b_id], trans_a, trans_b, trans_c)
        return self.add_op_with_args(op, [a_id, b_id])

    def flex_matmul(
        self,
        a_id,
        b_id,
        trans_a=False,
        trans_b=False,
        trans_c=False,
        n_batch_dims=0,
        n_reduce_dims=1,
        a_pivots=None,
        b_pivots=None,
        **attrs,
    ):
        op = FlexMatmulOp(
            self[a_id],
            self[b_id],
            trans_a,
            trans_b,
            trans_c,
            n_batch_dims,
            n_reduce_dims,
            a_pivots,
            b_pivots,
            **attrs,
        )
        return self.add_op_with_args(op, [a_id, b_id])

    def ewise_binop(self, op_name, a_id, b_id, commutative=False):
        op = EwiseBinOp(op_name, self[a_id], self[b_id], commutative)
        return self.add_op_with_args(op, [a_id, b_id])

    def add(self, a_id, b_id):
        return self.ewise_binop("add", a_id, b_id, True)

    def mul(self, a_id, b_id):
        return self.ewise_binop("mul", a_id, b_id, True)

    def sub(self, a_id, b_id):
        return self.ewise_binop("sub", a_id, b_id, False)

    def div(self, a_id, b_id):
        return self.ewise_binop("div", a_id, b_id, False)

    def ewise_uniop(self, op_name, x_id):
        op = EwiseUniOp(op_name, self[x_id])
        return self.add_op_with_args(op, [x_id])

    def relu(self, x_id):
        return self.ewise_uniop("relu", x_id)

    def tanh(self, x_id):
        return self.ewise_uniop("tanh", x_id)

    def sigmoid(self, x_id):
        return self.ewise_uniop("sigmoid", x_id)

    def neg(self, x_id):
        return self.ewise_uniop("neg", x_id)

    def reciprocal(self, x_id):
        return self.ewise_uniop("reciprocal", x_id)

    def exp(self, x_id):
        return self.ewise_uniop("exp", x_id)

    def log(self, x_id):
        return self.ewise_uniop("log", x_id)

    def ewise_uni_const(self, op_name, x_id, const):
        op = EwiseUniConstOp(op_name, self[x_id], const=const)
        return self.add_op_with_args(op, [x_id])

    def add_const(self, x_id, const):
        return self.ewise_uni_const("add", x_id, const)

    def mul_const(self, x_id, const):
        return self.ewise_uni_const("mul", x_id, const)

    def sub_const(self, x_id, const):
        return self.ewise_uni_const("sub", x_id, const)

    def div_const(self, x_id, const):
        return self.ewise_uni_const("div", x_id, const)

    def pow_const(self, x_id, const):
        return self.ewise_uni_const("pow", x_id, const)

    def reduce(self, op_name, x_id, dim, keepdim=False):
        op = ReduceOp(op_name, self[x_id], dim, keepdim)
        return self.add_op_with_args(op, [x_id])

    def sum(self, x_id, dim, keepdim=False):
        return self.reduce("sum", x_id, dim, keepdim)

    def max(self, x_id, dim, keepdim=False):
        return self.reduce("max", x_id, dim, keepdim)

    def min(self, x_id, dim, keepdim=False):
        return self.reduce("min", x_id, dim, keepdim)

    def mean(self, x_id, dim, keepdim=False):
        return self.reduce("mean", x_id, dim, keepdim)

    def softmax(self, x_id, dim):
        op = SoftmaxOp(self[x_id], dim)
        return self.add_op_with_args(op, [x_id])

    def concat(self, op_ids: List[OpId], dim):
        op = ConcatOp([self[i] for i in op_ids], dim)
        return self.add_op_with_args(op, op_ids)

    def slice(self, op_id: OpId, dim, start, end):
        op = SliceOp(self[op_id], dim, start, end)
        return self.add_op_with_args(op, [op_id])

    def split(self, op_id: OpId, dim, split_sizes: Union[int, List[int]]) -> List[OpId]:
        dim_size = self[op_id].out_shape[dim]
        if not isinstance(split_sizes, (list, tuple)):
            split_sizes = int(split_sizes)
            assert dim_size % split_sizes == 0
            split_sizes = [split_sizes] * (dim_size // split_sizes)
        else:
            assert sum(split_sizes) == dim_size

        start = 0
        op_ids = []
        for size in split_sizes:
            op_ids.append(self.slice(op_id, dim, start, start + size))
            start += size
        return op_ids

    def padding(self, op_id: OpId, pads: List[Tuple[int, int]]):
        op = PaddingOp(self[op_id], pads)
        return self.add_op_with_args(op, [op_id])

    def interpolate(self, op_id, size=None, scale=None, mode="nearest", layout=None):
        layout = layout or Operator.DEFAULT_IMAGE_LAYOUT
        op = InterpolateOp(self[op_id], size, scale, mode, layout)
        return self.add_op_with_args(op, [op_id])

    def reshape(self, x_id, shape):
        op = ReshapeOp(self[x_id], shape)
        return self.add_op_with_args(op, [x_id])

    def permute(self, x_id, dims):
        op = PermuteOp(self[x_id], dims)
        return self.add_op_with_args(op, [x_id])

    def attention(self, q_id, k_id, v_id):
        emb_dim = self[q_id].out_shape[-1]
        qk_id = self.matmul(q_id, k_id, trans_b=True)
        qk_id = self.div_const(qk_id, math.sqrt(emb_dim))
        qk_id = self.softmax(qk_id, -1)
        return self.matmul(qk_id, v_id)

    def load(self, op_id):
        op = LoadOp(self[op_id])
        return self.add_op_with_args(op, [op_id])

    def store(self, op_id):
        op = StoreOp(self[op_id])
        return self.add_op_with_args(op, [op_id])

    def Linear(
        self,
        inp_id,
        out_features,
        n_in_feat_dims=1,
        activation=None,
        weight_name_hint="w",
    ):
        in_features = self[inp_id].out_shape[-n_in_feat_dims:]
        out_features = utils.to_tuple(out_features)
        wgt_id = self.placeholder(
            [*in_features, *out_features], weight_name_hint, is_weight=True
        )
        ret = self.flex_matmul(
            inp_id, wgt_id, n_reduce_dims=n_in_feat_dims, activation=activation
        )
        if activation is not None:
            ret = self.ewise_uniop(activation, ret)
        return ret

    def Conv2d(
        self,
        img_id,
        out_channel,
        kernel_size,
        padding=0,
        stride=1,
        dilation=1,
        activation=None,
        layout=None,
        weight_name_hint="w",
    ):
        layout = layout or Operator.DEFAULT_IMAGE_LAYOUT
        n, c, h, w = utils.to_nchw(self[img_id].out_shape, layout)
        k = out_channel
        r, s = utils.to_nd_tuple(kernel_size)
        wgt_shape = utils.from_nchw((k, c, r, s), layout)
        wgt_id = self.placeholder(wgt_shape, weight_name_hint, is_weight=True)
        # print((n, h, c, k, r, stride, padding, dilation), ",")
        ret = self.conv2d(img_id, wgt_id, layout, padding, stride, dilation, activation)
        if activation is not None:
            ret = self.ewise_uniop(activation, ret)
        return ret

    def MultiheadAttention(self, q_id, k_id, v_id, emb_dim, num_heads=1):
        assert (
            len(set(tuple(self[x_id].out_shape[:2]) for x_id in [q_id, k_id, v_id]))
            == 1
        )
        batch, seq, *_ = self[q_id].out_shape
        if num_heads > 1:
            assert emb_dim % num_heads == 0
            head_dim = emb_dim // num_heads
            emb_ids = (
                self.Linear(x_id, [num_heads, head_dim]) for x_id in [q_id, k_id, v_id]
            )  # batch, seq, num_heads, head_dim
            emb_ids = (
                self.permute(x_id, [0, 2, 1, 3]) for x_id in emb_ids
            )  # batch, num_heads, seq, head_dim
            o_id = self.attention(*emb_ids)
            o_id = self.permute(o_id, [0, 2, 1, 3])
            # o_id = self.reshape(o_id, (batch, seq, emb_dim))
            # o_id = self.Linear(o_id, emb_dim)
            o_id = self.Linear(o_id, emb_dim, n_in_feat_dims=2)
        else:
            emb_ids = (
                self.Linear(x_id, emb_dim) for x_id in [q_id, k_id, v_id]
            )  # batch, seq, emb_dim
            assert num_heads == 1
            o_id = self.attention(*emb_ids)
            o_id = self.Linear(o_id, emb_dim)
        return o_id

    def LayerNorm(self, x_id, norm_shape):
        # we do not apply extra affine transform for LayerNorm in our experiments
        op = LayerNormOp(self[x_id], norm_shape)
        ys = [IndexOutOp(op, i, op._dim_links[0]) for i in range(3)]
        op_id = self.add_op_with_args(op, [x_id])
        return [self.add_op_with_args(y, [op_id]) for y in ys][0]

    # [END] operator builders
    # ----------------------------------------------------------------
