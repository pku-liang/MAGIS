from ..base import *


class MergeMatmulRule(RewriteRule):
    def _init_pattern_info(self) -> Tuple[OpGraph, List[OpId], List[OpId]]:
        P = OpGraph()
        x = P.placeholder([1, 1], "x")
        w1 = P.placeholder([1, 1], "w1")
        w2 = P.placeholder([1, 1], "w2")
        self.y1 = y1 = P.matmul(x, w1)
        self.y2 = y2 = P.matmul(x, w2)
        return P, [x, w1, w2], [y1, y2]

    def _create_new_subgraph(
        self, inp_ops: List[Operator], pms: MatchResult
    ) -> Tuple[OpGraph, List[OpId], List[OpId]]:
        gy1 = pms.ori_graph[pms.p2g_maps[self.y1]]
        gy2 = pms.ori_graph[pms.p2g_maps[self.y2]]
        if tuple(sorted(gy1.attrs.items())) != tuple(sorted(gy2.attrs.items())):
            return False
        db = int(not gy1.attrs["trans_b"]) - 2
        dc = int(not gy1.attrs["trans_c"]) - 2
        S = OpGraph()
        x, w1, w2 = S.add_nodes(inp_ops)
        w = S.concat([w1, w2], db)
        y = S.matmul(x, w, **gy1.attrs)
        y1, y2 = S.split(y, dc, [S[w1].out_shape[db], S[w2].out_shape[db]])
        return S, [x, w1, w2], [y1, y2]


class MergeMatmulRule1(RewriteRule):
    def _init_pattern_info(self) -> Tuple[OpGraph, List[OpId], List[OpId]]:
        P = OpGraph()
        w = P.placeholder([1, 1])
        x1 = P.placeholder([1, 1])
        x2 = P.placeholder([1, 1])
        self.y1 = y1 = P.matmul(x1, w)
        self.y2 = y2 = P.matmul(x2, w)
        return P, [x1, x2, w], [y1, y2]

    def _create_new_subgraph(
        self, inp_ops: List[Operator], pms: MatchResult
    ) -> Tuple[OpGraph, List[OpId], List[OpId]]:
        gy1 = pms.ori_graph[pms.p2g_maps[self.y1]]
        gy2 = pms.ori_graph[pms.p2g_maps[self.y2]]
        if tuple(sorted(gy1.attrs.items())) != tuple(sorted(gy2.attrs.items())):
            return False
        da = int(not gy1.attrs["trans_a"]) - 2
        dc = int(not gy1.attrs["trans_c"]) - 2
        S = OpGraph()
        w, x1, x2 = S.add_nodes(inp_ops)
        x = S.concat([x1, x2], da)
        y = S.matmul(x, w, **gy1.attrs)
        y1, y2 = S.split(y, dc, [S[x1].out_shape[da], S[x2].out_shape[da]])
        return S, [x1, x2, w], [y1, y2]


class MergeFlexMatmulRule(RewriteRule):
    def _init_pattern_info(self) -> Tuple[OpGraph, List[OpId], List[OpId]]:
        P = OpGraph()
        x = P.placeholder([1, 1], "x")
        w1 = P.placeholder([1, 1], "w1")
        w2 = P.placeholder([1, 1], "w2")
        self.y1 = y1 = P.flex_matmul(x, w1)
        self.y2 = y2 = P.flex_matmul(x, w2)
        return P, [x, w1, w2], [y1, y2]

    def _create_new_subgraph(
        self, inp_ops: List[Operator], pms: MatchResult
    ) -> Tuple[OpGraph, List[OpId], List[OpId]]:
        gy1 = pms.ori_graph[pms.p2g_maps[self.y1]]
        gy2 = pms.ori_graph[pms.p2g_maps[self.y2]]
        if tuple(sorted(gy1.attrs.items())) != tuple(sorted(gy2.attrs.items())):
            return False
        ws1 = gy1.inp_shapes[1]
        ws2 = gy2.inp_shapes[1]
        if len(ws1) != len(ws2):
            return False

        nb = gy1.attrs["n_batch_dims"]
        nr = gy1.attrs["n_reduce_dims"]
        nn = len(ws1) - nb - nr
        nm = len(gy1.out_shape) - nb - nn

        db0 = nb if gy1.attrs["trans_b"] else nb + nr
        db1 = db0 + nn
        db = -1
        for i in range(db0, db1):
            if ws1[i] != ws2[i]:
                assert db == -1
                db = i
        if db == -1:
            db = db0
        dc = (nb if gy1.attrs["trans_c"] else nb + nm) + (db - db0)

        S = OpGraph()
        x, w1, w2 = S.add_nodes(inp_ops)
        w = S.concat([w1, w2], db)
        y = S.flex_matmul(x, w, **gy1.attrs)
        y1, y2 = S.split(y, dc, [S[w1].out_shape[db], S[w2].out_shape[db]])
        return S, [x, w1, w2], [y1, y2]


# class MergeReluRule(RewriteRule):
#     def _init_pattern_info(self) -> Tuple[Graph, List[OpId], List[OpId]]:
#         P = Graph()
#         x1 = P.placeholder([1])
#         x2 = P.placeholder([1])
#         y1 = P.ewise_uniop("relu", x1)
#         y2 = P.ewise_uniop("relu", x2)
#         return P, [x1, x2], [y1, y2]

#     def _create_new_subgraph(
#         self, inp_ops: List[Operator], pms: MatchResult
#     ) -> Tuple[Graph, List[OpId], List[OpId]]:
#         x1, x2 = inp_ops
#         s1, s2 = x1.out_shape, x2.out_shape
#         if len(s1) != len(s2):
#             return False
#         fst_neq_dim = next((i for i, (l1, l2) in enumerate(zip(s1, s2)) if l1 != l2), 0)
#         if tuple(s1[fst_neq_dim + 1 :]) != tuple(s2[fst_neq_dim + 1 :]):
#             return False
#         S = Graph()
#         x1, x2 = S.add_ops(inp_ops)
#         x = S.concat([x1, x2], fst_neq_dim)
#         y = S.ewise_uniop("relu", x)
#         y1, y2 = S.split(y, fst_neq_dim, [s1[fst_neq_dim], s2[fst_neq_dim]])
#         return S, [x1, x2], [y1, y2]


class MergeConv2dRule(RewriteRule):
    def _init_pattern_info(self) -> Tuple[OpGraph, List[OpId], List[OpId]]:
        P = OpGraph()
        x = P.placeholder([1, 1, 1, 1])
        w1 = P.placeholder([1, 1, 1, 1])
        w2 = P.placeholder([1, 1, 1, 1])
        self.y1 = y1 = P.conv2d(x, w1)
        self.y2 = y2 = P.conv2d(x, w2)
        return P, [x, w1, w2], [y1, y2]

    def _create_new_subgraph(
        self, inp_ops: List[Operator], pms: MatchResult
    ) -> Tuple[OpGraph, List[OpId], List[OpId]]:
        gy1 = pms.ori_graph[pms.p2g_maps[self.y1]]
        gy2 = pms.ori_graph[pms.p2g_maps[self.y2]]
        if tuple(sorted(gy1.attrs.items())) != tuple(sorted(gy2.attrs.items())):
            return False
        S = OpGraph()
        x, w1, w2 = S.add_nodes(inp_ops)

        # try padding if kernel_sizes are different
        # corresponding to TASO's "enlarge" (padding) rules
        layout = gy1.attrs["layout"]
        slc = slice(-2, None) if layout == "nchw" else slice(1, -1)
        r0, s0 = [
            max(s1, s2) for s1, s2 in zip(S[w1].out_shape[slc], S[w2].out_shape[slc])
        ]

        def pad(w):
            shp = S[w].out_shape
            r, s = shp[slc]
            pr = ((r0 - r) // 2, (r0 - r + 1) // 2)
            ps = ((s0 - s) // 2, (s0 - s + 1) // 2)
            if sum(pr) + sum(ps) == 0:
                return w
            pads = [0] * len(shp)
            pads[slc] = pr, ps
            return S.padding(w, pads)

        w1, w2 = pad(w1), pad(w2)

        w = S.concat([w1, w2], 0)
        y = S.conv2d(x, w, **gy1.attrs)
        y1, y2 = S.split(
            y,
            1 if layout == "nchw" else 4,
            [S[w1].out_shape[0], S[w2].out_shape[0]],
        )
        return S, [x, w1, w2], [y1, y2]


class MergeConv2dRule1(RewriteRule):
    def _init_pattern_info(self) -> Tuple[OpGraph, List[OpId], List[OpId]]:
        P = OpGraph()
        x1 = P.placeholder([1, 1, 1, 1])
        x2 = P.placeholder([1, 1, 1, 1])
        w = P.placeholder([1, 1, 1, 1])
        self.y1 = y1 = P.conv2d(x1, w)
        self.y2 = y2 = P.conv2d(x2, w)
        return P, [x1, x2, w], [y1, y2]

    def _create_new_subgraph(
        self, inp_ops: List[Operator], pms: MatchResult
    ) -> Tuple[OpGraph, List[OpId], List[OpId]]:
        gy1 = pms.ori_graph[pms.p2g_maps[self.y1]]
        gy2 = pms.ori_graph[pms.p2g_maps[self.y2]]
        if tuple(sorted(gy1.attrs.items())) != tuple(sorted(gy2.attrs.items())):
            return False
        S = OpGraph()
        x1, x2, w = S.add_nodes(inp_ops)
        x = S.concat([x1, x2], 0)
        y = S.conv2d(x, w, **gy1.attrs)
        y1, y2 = S.split(y, 0, [S[x1].out_shape[0], S[x2].out_shape[0]])
        return S, [x1, x2, w], [y1, y2]
