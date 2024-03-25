from .base import *


class SwappingRule(RewriteRule):
    def _init_pattern_info(self) -> Tuple[OpGraph, List[OpId], List[OpId]]:
        P = OpGraph()
        x = P.placeholder([1])
        y = P.relu(x)

        """
        The returned input_ids and output_ids technically have no distinction... 
        Their union represents the nodes in the original sub-graph that are connected to outside. 
        The only difference is that input-ops can be used directly within _create_new_subgraph. 
        Due to some historical reasons and it's not convenient to modify such interface.
        """
        return P, [x, y], []

    def _match_edge(self, g_etag: ArgIndex, p_etag: ArgIndex):
        return True

    def _match_op(self, g_op: Operator, p_op: Operator):
        return not g_op.is_mem_op()

    def _create_new_subgraph(
        self, inp_ops: List[Operator], pms: MatchResult
    ) -> Tuple[OpGraph, List[OpId], List[OpId]]:
        gx, gy = inp_ops
        gx_id, gy_id = pms.g_inp_ids
        arg_idx = pms.ori_graph.etag_between(gx_id, gy_id)
        S = OpGraph()
        x = S.add_node(gx)
        y = S.store(x)
        y = S.load(y)
        y = S.add_op_with_args(gy, [y], [arg_idx])
        return S, [x, y], []


# recompute for unary-op
class RecomputeRule1(RewriteRule):
    def _init_pattern_info(self) -> Tuple[OpGraph, List[OpId], List[OpId]]:
        P = OpGraph()
        x = P.placeholder([1])
        y = P.relu(x)
        z1 = P.relu(y)
        z2 = P.relu(y)
        self.y_op = P[y]
        return P, [x, y, z1, z2], []

    def _match_edge(self, g_etag: ArgIndex, p_etag: ArgIndex):
        return True

    def _match_op(self, g_op: Operator, p_op: Operator):
        return not (self.y_op is p_op and len(g_op.inp_shapes) != 1)

    def _create_new_subgraph(
        self, inp_ops: List[Operator], pms: MatchResult
    ) -> Tuple[OpGraph, List[OpId], List[OpId]]:
        gx, gy, gz1, gz2 = inp_ops
        _, gy_id, gz1_id, gz2_id = pms.g_inp_ids
        arg_idx1 = pms.ori_graph.etag_between(gy_id, gz1_id)
        arg_idx2 = pms.ori_graph.etag_between(gy_id, gz2_id)
        S = OpGraph()
        x = S.add_node(gx)
        y1 = S.add_op_with_args(gy, [x])
        y2 = S.add_op_with_args(gy, [x])
        z1 = S.add_op_with_args(gz1, [y1], [arg_idx1])
        z2 = S.add_op_with_args(gz2, [y2], [arg_idx2])
        return S, [x, y1, z1, z2], []


# recompute for binary-op
class RecomputeRule2(RewriteRule):
    def _init_pattern_info(self) -> Tuple[OpGraph, List[OpId], List[OpId]]:
        P = OpGraph()
        x1 = P.placeholder([1])
        x2 = P.placeholder([1])
        self.y = y = P.add(x1, x2)
        self.z1 = z1 = P.relu(y)
        self.z2 = z2 = P.relu(y)
        self.y_op = P[y]
        return P, [x1, x2, y, z1, z2], []

    def _match_edge(self, g_etag: ArgIndex, p_etag: ArgIndex):
        return True

    def _match_op(self, g_op: Operator, p_op: Operator):
        return not (self.y_op is p_op and len(g_op.inp_shapes) != 2)

    def _create_new_subgraph(
        self, inp_ops: List[Operator], pms: MatchResult
    ) -> Tuple[OpGraph, List[OpId], List[OpId]]:
        *gxs, gy, gz1, gz2 = inp_ops
        *_, gy_id, gz1_id, gz2_id = pms.g_inp_ids
        arg_idx1 = pms.ori_graph.etag_between(gy_id, gz1_id)
        arg_idx2 = pms.ori_graph.etag_between(gy_id, gz2_id)
        S = OpGraph()
        x1, x2 = S.add_nodes(gxs)
        y1 = S.add_op_with_args(gy, [x1, x2])
        y2 = S.add_op_with_args(gy, [x1, x2])
        z1 = S.add_op_with_args(gz1, [y1], [arg_idx1])
        z2 = S.add_op_with_args(gz2, [y2], [arg_idx2])
        return S, [x1, x2, y1, z1, z2], []


class DeSwappingRule1(RewriteRule):
    def _init_pattern_info(self) -> Tuple[OpGraph, List[OpId], List[OpId]]:
        P = OpGraph()
        x = P.placeholder([1])
        y = P.store(x)
        y = P.load(y)
        return P, [x], [y]

    def _create_new_subgraph(
        self, inp_ops: List[Operator], pms: MatchResult
    ) -> Tuple[OpGraph, List[OpId], List[OpId]]:
        S = OpGraph()
        (x,) = S.add_nodes(inp_ops)
        return S, [x], [x]


# can be derived from recompute-rule + de-swapping-rule1
class DeSwappingRule2(RewriteRule):
    def _init_pattern_info(self) -> Tuple[OpGraph, List[OpId], List[OpId]]:
        P = OpGraph()
        x = P.placeholder([1])
        y = P.store(x)
        y1 = P.load(y)
        y2 = P.load(y)
        return P, [x], [y1, y2]

    def _create_new_subgraph(
        self, inp_ops: List[Operator], pms: MatchResult
    ) -> Tuple[OpGraph, List[OpId], List[OpId]]:
        S = OpGraph()
        (x,) = S.add_nodes(inp_ops)
        return S, [x], [x, x]


# can be derived from recompute-rule + de-swapping-rule1
class DeSwappingRule3(RewriteRule):
    def _init_pattern_info(self) -> Tuple[OpGraph, List[OpId], List[OpId]]:
        P = OpGraph()
        x = P.placeholder([1])
        y = P.store(x)
        y1 = P.load(y)
        y2 = P.load(y)
        y3 = P.load(y)
        return P, [x], [y1, y2, y3]

    def _create_new_subgraph(
        self, inp_ops: List[Operator], pms: MatchResult
    ) -> Tuple[OpGraph, List[OpId], List[OpId]]:
        S = OpGraph()
        (x,) = S.add_nodes(inp_ops)
        return S, [x], [x, x, x]


# de-recompute for unary-op
class DeRecomputeRule1(RewriteRule):
    def _init_pattern_info(self) -> Tuple[OpGraph, List[OpId], List[OpId]]:
        P = OpGraph()
        x = P.placeholder([1])
        y1 = P.relu(x)
        y2 = P.relu(x)
        return P, [x, y1, y2], []

    def _match_op(self, g_op: Operator, p_op: Operator):
        return p_op.tag == "placeholder" or len(g_op.inp_shapes) == len(p_op.inp_shapes)

    def _create_new_subgraph(
        self, inp_ops: List[Operator], pms: MatchResult
    ) -> Tuple[OpGraph, List[OpId], List[OpId]]:
        gx, gy1, gy2 = inp_ops
        if gy1.key != gy2.key:
            return False
        S = OpGraph()
        x = S.add_node(gx)
        y = S.add_op_with_args(gy1, [x])
        return S, [x, y, y], []


# de-recompute for binary-op
class DeRecomputeRule2(RewriteRule):
    def _init_pattern_info(self) -> Tuple[OpGraph, List[OpId], List[OpId]]:
        P = OpGraph()
        x1 = P.placeholder([1])
        x2 = P.placeholder([1])
        y1 = P.add(x1, x2)
        y2 = P.add(x1, x2)
        return P, [x1, x2, y1, y2], []

    def _match_op(self, g_op: Operator, p_op: Operator):
        return p_op.tag == "placeholder" or len(g_op.inp_shapes) == len(p_op.inp_shapes)

    def _create_new_subgraph(
        self, inp_ops: List[Operator], pms: MatchResult
    ) -> Tuple[OpGraph, List[OpId], List[OpId]]:
        *gxs, gy1, gy2 = inp_ops
        if gy1.key != gy2.key:
            return False
        S = OpGraph()
        x1, x2 = S.add_nodes(gxs)
        y = S.add_op_with_args(gy1, [x1, x2])
        return S, [x1, x2, y, y], []
