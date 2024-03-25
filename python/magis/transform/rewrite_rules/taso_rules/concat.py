from ..base import *


class ConcatAddRule(RewriteRule):
    def _init_pattern_info(self) -> Tuple[OpGraph, List[OpId], List[OpId]]:
        P = OpGraph()
        a, b, c = [P.placeholder([1]) for _ in range(3)]
        ab = P.add(a, b)
        bc = P.add(b, c)
        self.y = y = P.concat([ab, bc], 0)
        return P, [a, b, c], [y]

    def _create_new_subgraph(
        self, inp_ops: List[Operator], pms: MatchResult
    ) -> Tuple[OpGraph, List[OpId], List[OpId]]:
        S = OpGraph()
        gy = pms.ori_graph[pms.p2g_maps[self.y]]
        dim = gy.attrs["dim"]
        a, b, c = S.add_nodes(inp_ops)
        ab = S.concat([a, b], dim)
        bc = S.concat([b, c], dim)
        y = S.add(ab, bc)
        return S, [a, b, c], [y]


class ConcatAddRule1(RewriteRule):
    def _init_pattern_info(self) -> Tuple[OpGraph, List[OpId], List[OpId]]:
        P = OpGraph()
        a, b, c, d = [P.placeholder([1]) for _ in range(4)]
        ab = P.add(a, b)
        cd = P.add(c, d)
        self.y = y = P.concat([ab, cd], 0)
        return P, [a, b, c, d], [y]

    def _create_new_subgraph(
        self, inp_ops: List[Operator], pms: MatchResult
    ) -> Tuple[OpGraph, List[OpId], List[OpId]]:
        S = OpGraph()
        gy = pms.ori_graph[pms.p2g_maps[self.y]]
        dim = gy.attrs["dim"]
        a, b, c, d = S.add_nodes(inp_ops)
        ac = S.concat([a, c], dim)
        bd = S.concat([b, d], dim)
        y = S.add(ac, bd)
        return S, [a, b, c, d], [y]


class ConcatMatmulRule(RewriteRule):
    def _init_pattern_info(self) -> Tuple[OpGraph, List[OpId], List[OpId]]:
        P = OpGraph()
        x = P.placeholder([1, 1], "x")
        w1 = P.placeholder([1, 1], "w1")
        w2 = P.placeholder([1, 1], "w2")
        self.y1 = y1 = P.matmul(x, w1)
        self.y2 = y2 = P.matmul(x, w2)
        y = P.concat([y1, y2], 0)
        return P, [x, w1, w2], [y]

    def _create_new_subgraph(
        self, inp_ops: List[Operator], pms: MatchResult
    ) -> Tuple[OpGraph, List[OpId], List[OpId]]:
        gy1 = pms.ori_graph[pms.p2g_maps[self.y1]]
        gy2 = pms.ori_graph[pms.p2g_maps[self.y2]]
        if tuple(sorted(gy1.attrs.items())) != tuple(sorted(gy2.attrs.items())):
            return False
        db = int(not gy1.attrs["trans_b"])
        S = OpGraph()
        x, w1, w2 = S.add_nodes(inp_ops)
        w = S.concat([w1, w2], db)
        y = S.matmul(x, w, **gy1.attrs)
        return S, [x, w1, w2], [y]


class ConcatMatmulRule1(RewriteRule):
    def _init_pattern_info(self) -> Tuple[OpGraph, List[OpId], List[OpId]]:
        P = OpGraph()
        w = P.placeholder([1, 1])
        x1 = P.placeholder([1, 1])
        x2 = P.placeholder([1, 1])
        self.y1 = y1 = P.matmul(x1, w)
        self.y2 = y2 = P.matmul(x2, w)
        y = P.concat([y1, y2], 0)
        return P, [x1, x2, w], [y]

    def _create_new_subgraph(
        self, inp_ops: List[Operator], pms: MatchResult
    ) -> Tuple[OpGraph, List[OpId], List[OpId]]:
        gy1 = pms.ori_graph[pms.p2g_maps[self.y1]]
        gy2 = pms.ori_graph[pms.p2g_maps[self.y2]]
        if tuple(sorted(gy1.attrs.items())) != tuple(sorted(gy2.attrs.items())):
            return False
        da = int(not gy1.attrs["trans_a"])
        S = OpGraph()
        w, x1, x2 = S.add_nodes(inp_ops)
        x = S.concat([x1, x2], da)
        y = S.matmul(x, w, **gy1.attrs)
        return S, [x1, x2, w], [y]


class ConcatConv2dRule(RewriteRule):
    def _init_pattern_info(self) -> Tuple[OpGraph, List[OpId], List[OpId]]:
        P = OpGraph()
        x = P.placeholder([1, 1, 1, 1])
        w1 = P.placeholder([1, 1, 1, 1])
        w2 = P.placeholder([1, 1, 1, 1])
        self.y1 = y1 = P.conv2d(x, w1)
        self.y2 = y2 = P.conv2d(x, w2)
        y = P.concat([y1, y2], 0)
        return P, [x, w1, w2], [y]

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
        return S, [x, w1, w2], [y]


class ConcatConv2dRule1(RewriteRule):
    def _init_pattern_info(self) -> Tuple[OpGraph, List[OpId], List[OpId]]:
        P = OpGraph()
        x1 = P.placeholder([1, 1, 1, 1])
        x2 = P.placeholder([1, 1, 1, 1])
        w = P.placeholder([1, 1, 1, 1])
        self.y1 = y1 = P.conv2d(x1, w)
        self.y2 = y2 = P.conv2d(x2, w)
        y = P.concat([y1, y2], 0)
        return P, [x1, x2, w], [y]

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
        return S, [x1, x2, w], [y]


