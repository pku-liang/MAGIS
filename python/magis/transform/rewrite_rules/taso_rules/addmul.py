from ..base import *


class AddMulRule0(RewriteRule):
    def _init_pattern_info(self) -> Tuple[OpGraph, List[OpId], List[OpId]]:
        G = OpGraph()
        x13 = G.placeholder([1])
        x11 = G.placeholder([1])
        x10 = G.placeholder([1])
        y0 = G.add(x10, x11)
        y1 = G.add(x13, y0)
        y2 = G.add(x13, y1)
        return G, [x13, x11, x10], [y2]

    def _create_new_subgraph(
        self, inp_ops: List[Operator], pms: MatchResult
    ) -> Tuple[OpGraph, List[OpId], List[OpId]]:
        G = OpGraph()
        x13, x11, x10 = G.add_nodes(inp_ops)
        y0 = G.add(x10, x13)
        y1 = G.add(x11, x13)
        y2 = G.add(y0, y1)
        return G, [x13, x11, x10], [y2]


class AddMulRule1(RewriteRule):
    def _init_pattern_info(self) -> Tuple[OpGraph, List[OpId], List[OpId]]:
        G = OpGraph()
        x13 = G.placeholder([1])
        x11 = G.placeholder([1])
        x10 = G.placeholder([1])
        y0 = G.add(x10, x11)
        y1 = G.add(x13, y0)
        return G, [x13, x11, x10], [y1]

    def _create_new_subgraph(
        self, inp_ops: List[Operator], pms: MatchResult
    ) -> Tuple[OpGraph, List[OpId], List[OpId]]:
        G = OpGraph()
        x13, x11, x10 = G.add_nodes(inp_ops)
        y0 = G.add(x10, x13)
        y1 = G.add(x11, y0)
        return G, [x13, x11, x10], [y1]


class AddMulRule2(RewriteRule):
    def _init_pattern_info(self) -> Tuple[OpGraph, List[OpId], List[OpId]]:
        G = OpGraph()
        x13 = G.placeholder([1])
        x11 = G.placeholder([1])
        x10 = G.placeholder([1])
        y0 = G.add(x10, x11)
        y1 = G.add(x11, x13)
        y2 = G.add(y0, y1)
        return G, [x13, x11, x10], [y2]

    def _create_new_subgraph(
        self, inp_ops: List[Operator], pms: MatchResult
    ) -> Tuple[OpGraph, List[OpId], List[OpId]]:
        G = OpGraph()
        x13, x11, x10 = G.add_nodes(inp_ops)
        y0 = G.add(x10, x13)
        y1 = G.add(x11, y0)
        y2 = G.add(x11, y1)
        return G, [x13, x11, x10], [y2]


class AddMulRule3(RewriteRule):
    def _init_pattern_info(self) -> Tuple[OpGraph, List[OpId], List[OpId]]:
        G = OpGraph()
        x13 = G.placeholder([1])
        x11 = G.placeholder([1])
        x10 = G.placeholder([1])
        y0 = G.add(x10, x11)
        y1 = G.add(x13, y0)
        y2 = G.add(x13, y1)
        return G, [x13, x11, x10], [y2]

    def _create_new_subgraph(
        self, inp_ops: List[Operator], pms: MatchResult
    ) -> Tuple[OpGraph, List[OpId], List[OpId]]:
        G = OpGraph()
        x13, x11, x10 = G.add_nodes(inp_ops)
        y0 = G.add(x10, x13)
        y1 = G.add(x13, y0)
        y2 = G.add(x11, y1)
        return G, [x13, x11, x10], [y2]


class AddMulRule4(RewriteRule):
    def _init_pattern_info(self) -> Tuple[OpGraph, List[OpId], List[OpId]]:
        G = OpGraph()
        x13 = G.placeholder([1])
        x11 = G.placeholder([1])
        x10 = G.placeholder([1])
        y0 = G.add(x10, x11)
        y1 = G.add(x13, y0)
        y2 = G.add(x13, y1)
        return G, [x13, x11, x10], [y2]

    def _create_new_subgraph(
        self, inp_ops: List[Operator], pms: MatchResult
    ) -> Tuple[OpGraph, List[OpId], List[OpId]]:
        G = OpGraph()
        x13, x11, x10 = G.add_nodes(inp_ops)
        y0 = G.add(x11, x13)
        y1 = G.add(x10, x13)
        y2 = G.add(y0, y1)
        return G, [x13, x11, x10], [y2]


class AddMulRule5(RewriteRule):
    def _init_pattern_info(self) -> Tuple[OpGraph, List[OpId], List[OpId]]:
        G = OpGraph()
        x13 = G.placeholder([1])
        x11 = G.placeholder([1])
        x10 = G.placeholder([1])
        y0 = G.add(x10, x11)
        y1 = G.add(x13, y0)
        return G, [x13, x11, x10], [y1]

    def _create_new_subgraph(
        self, inp_ops: List[Operator], pms: MatchResult
    ) -> Tuple[OpGraph, List[OpId], List[OpId]]:
        G = OpGraph()
        x13, x11, x10 = G.add_nodes(inp_ops)
        y0 = G.add(x11, x13)
        y1 = G.add(x10, y0)
        return G, [x13, x11, x10], [y1]


class AddMulRule6(RewriteRule):
    def _init_pattern_info(self) -> Tuple[OpGraph, List[OpId], List[OpId]]:
        G = OpGraph()
        x13 = G.placeholder([1])
        x11 = G.placeholder([1])
        x10 = G.placeholder([1])
        y0 = G.add(x10, x11)
        y1 = G.add(x10, x13)
        y2 = G.add(y0, y1)
        return G, [x13, x11, x10], [y2]

    def _create_new_subgraph(
        self, inp_ops: List[Operator], pms: MatchResult
    ) -> Tuple[OpGraph, List[OpId], List[OpId]]:
        G = OpGraph()
        x13, x11, x10 = G.add_nodes(inp_ops)
        y0 = G.add(x11, x13)
        y1 = G.add(x10, y0)
        y2 = G.add(x10, y1)
        return G, [x13, x11, x10], [y2]


class AddMulRule7(RewriteRule):
    def _init_pattern_info(self) -> Tuple[OpGraph, List[OpId], List[OpId]]:
        G = OpGraph()
        x13 = G.placeholder([1])
        x11 = G.placeholder([1])
        x10 = G.placeholder([1])
        y0 = G.add(x10, x11)
        y1 = G.add(x13, y0)
        y2 = G.add(x13, y1)
        return G, [x13, x11, x10], [y2]

    def _create_new_subgraph(
        self, inp_ops: List[Operator], pms: MatchResult
    ) -> Tuple[OpGraph, List[OpId], List[OpId]]:
        G = OpGraph()
        x13, x11, x10 = G.add_nodes(inp_ops)
        y0 = G.add(x11, x13)
        y1 = G.add(x13, y0)
        y2 = G.add(x10, y1)
        return G, [x13, x11, x10], [y2]


class AddMulRule8(RewriteRule):
    def _init_pattern_info(self) -> Tuple[OpGraph, List[OpId], List[OpId]]:
        G = OpGraph()
        x13 = G.placeholder([1])
        x11 = G.placeholder([1])
        x10 = G.placeholder([1])
        y0 = G.add(x10, x11)
        y1 = G.mul(x13, y0)
        return G, [x13, x11, x10], [y1]

    def _create_new_subgraph(
        self, inp_ops: List[Operator], pms: MatchResult
    ) -> Tuple[OpGraph, List[OpId], List[OpId]]:
        G = OpGraph()
        x13, x11, x10 = G.add_nodes(inp_ops)
        y0 = G.mul(x13, x10)
        y1 = G.mul(x13, x11)
        y2 = G.add(y0, y1)
        return G, [x13, x11, x10], [y2]


class AddMulRule9(RewriteRule):
    def _init_pattern_info(self) -> Tuple[OpGraph, List[OpId], List[OpId]]:
        G = OpGraph()
        x13 = G.placeholder([1])
        x11 = G.placeholder([1])
        x10 = G.placeholder([1])
        y0 = G.add(x10, x11)
        y1 = G.mul(x13, y0)
        return G, [x13, x11, x10], [y1]

    def _create_new_subgraph(
        self, inp_ops: List[Operator], pms: MatchResult
    ) -> Tuple[OpGraph, List[OpId], List[OpId]]:
        G = OpGraph()
        x13, x11, x10 = G.add_nodes(inp_ops)
        y0 = G.mul(x10, x13)
        y1 = G.mul(x13, x11)
        y2 = G.add(y0, y1)
        return G, [x13, x11, x10], [y2]


class AddMulRule10(RewriteRule):
    def _init_pattern_info(self) -> Tuple[OpGraph, List[OpId], List[OpId]]:
        G = OpGraph()
        x13 = G.placeholder([1])
        x11 = G.placeholder([1])
        x10 = G.placeholder([1])
        y0 = G.mul(x10, x11)
        y1 = G.mul(x13, y0)
        y2 = G.add(y0, y1)
        return G, [x13, x11, x10], [y2]

    def _create_new_subgraph(
        self, inp_ops: List[Operator], pms: MatchResult
    ) -> Tuple[OpGraph, List[OpId], List[OpId]]:
        G = OpGraph()
        x13, x11, x10 = G.add_nodes(inp_ops)
        y0 = G.mul(x10, x13)
        y1 = G.add(x10, y0)
        y2 = G.mul(x11, y1)
        return G, [x13, x11, x10], [y2]


class AddMulRule11(RewriteRule):
    def _init_pattern_info(self) -> Tuple[OpGraph, List[OpId], List[OpId]]:
        G = OpGraph()
        x14 = G.placeholder([1])
        x11 = G.placeholder([1])
        x10 = G.placeholder([1])
        y0 = G.mul(x10, x11)
        y1 = G.add(x11, y0)
        y2 = G.mul(x14, y1)
        return G, [x14, x11, x10], [y2]

    def _create_new_subgraph(
        self, inp_ops: List[Operator], pms: MatchResult
    ) -> Tuple[OpGraph, List[OpId], List[OpId]]:
        G = OpGraph()
        x14, x11, x10 = G.add_nodes(inp_ops)
        y0 = G.mul(x10, x14)
        y1 = G.add(x14, y0)
        y2 = G.mul(x11, y1)
        return G, [x14, x11, x10], [y2]


class AddMulRule12(RewriteRule):
    def _init_pattern_info(self) -> Tuple[OpGraph, List[OpId], List[OpId]]:
        G = OpGraph()
        x13 = G.placeholder([1])
        x11 = G.placeholder([1])
        x10 = G.placeholder([1])
        y0 = G.add(x10, x11)
        y1 = G.mul(x13, y0)
        return G, [x13, x11, x10], [y1]

    def _create_new_subgraph(
        self, inp_ops: List[Operator], pms: MatchResult
    ) -> Tuple[OpGraph, List[OpId], List[OpId]]:
        G = OpGraph()
        x13, x11, x10 = G.add_nodes(inp_ops)
        y0 = G.mul(x10, x13)
        y1 = G.mul(x11, x13)
        y2 = G.add(y0, y1)
        return G, [x13, x11, x10], [y2]


class AddMulRule13(RewriteRule):
    def _init_pattern_info(self) -> Tuple[OpGraph, List[OpId], List[OpId]]:
        G = OpGraph()
        x13 = G.placeholder([1])
        x11 = G.placeholder([1])
        x10 = G.placeholder([1])
        y0 = G.mul(x10, x11)
        y1 = G.mul(x13, y0)
        y2 = G.mul(x13, y1)
        return G, [x13, x11, x10], [y2]

    def _create_new_subgraph(
        self, inp_ops: List[Operator], pms: MatchResult
    ) -> Tuple[OpGraph, List[OpId], List[OpId]]:
        G = OpGraph()
        x13, x11, x10 = G.add_nodes(inp_ops)
        y0 = G.mul(x10, x13)
        y1 = G.mul(x11, x13)
        y2 = G.mul(y0, y1)
        return G, [x13, x11, x10], [y2]


class AddMulRule14(RewriteRule):
    def _init_pattern_info(self) -> Tuple[OpGraph, List[OpId], List[OpId]]:
        G = OpGraph()
        x13 = G.placeholder([1])
        x11 = G.placeholder([1])
        x10 = G.placeholder([1])
        y0 = G.mul(x10, x11)
        y1 = G.mul(x13, y0)
        return G, [x13, x11, x10], [y1]

    def _create_new_subgraph(
        self, inp_ops: List[Operator], pms: MatchResult
    ) -> Tuple[OpGraph, List[OpId], List[OpId]]:
        G = OpGraph()
        x13, x11, x10 = G.add_nodes(inp_ops)
        y0 = G.mul(x10, x13)
        y1 = G.mul(x11, y0)
        return G, [x13, x11, x10], [y1]


class AddMulRule15(RewriteRule):
    def _init_pattern_info(self) -> Tuple[OpGraph, List[OpId], List[OpId]]:
        G = OpGraph()
        x13 = G.placeholder([1])
        x11 = G.placeholder([1])
        x10 = G.placeholder([1])
        y0 = G.mul(x10, x11)
        y1 = G.mul(x11, x13)
        y2 = G.mul(y0, y1)
        return G, [x13, x11, x10], [y2]

    def _create_new_subgraph(
        self, inp_ops: List[Operator], pms: MatchResult
    ) -> Tuple[OpGraph, List[OpId], List[OpId]]:
        G = OpGraph()
        x13, x11, x10 = G.add_nodes(inp_ops)
        y0 = G.mul(x10, x13)
        y1 = G.mul(x11, y0)
        y2 = G.mul(x11, y1)
        return G, [x13, x11, x10], [y2]


class AddMulRule16(RewriteRule):
    def _init_pattern_info(self) -> Tuple[OpGraph, List[OpId], List[OpId]]:
        G = OpGraph()
        x13 = G.placeholder([1])
        x11 = G.placeholder([1])
        x10 = G.placeholder([1])
        y0 = G.mul(x10, x11)
        y1 = G.mul(x13, y0)
        y2 = G.mul(x13, y1)
        return G, [x13, x11, x10], [y2]

    def _create_new_subgraph(
        self, inp_ops: List[Operator], pms: MatchResult
    ) -> Tuple[OpGraph, List[OpId], List[OpId]]:
        G = OpGraph()
        x13, x11, x10 = G.add_nodes(inp_ops)
        y0 = G.mul(x10, x13)
        y1 = G.mul(x13, y0)
        y2 = G.mul(x11, y1)
        return G, [x13, x11, x10], [y2]


class AddMulRule17(RewriteRule):
    def _init_pattern_info(self) -> Tuple[OpGraph, List[OpId], List[OpId]]:
        G = OpGraph()
        x13 = G.placeholder([1])
        x11 = G.placeholder([1])
        x10 = G.placeholder([1])
        y0 = G.mul(x10, x11)
        y1 = G.mul(x13, y0)
        y2 = G.add(y0, y1)
        return G, [x13, x11, x10], [y2]

    def _create_new_subgraph(
        self, inp_ops: List[Operator], pms: MatchResult
    ) -> Tuple[OpGraph, List[OpId], List[OpId]]:
        G = OpGraph()
        x13, x11, x10 = G.add_nodes(inp_ops)
        y0 = G.mul(x11, x13)
        y1 = G.add(x11, y0)
        y2 = G.mul(x10, y1)
        return G, [x13, x11, x10], [y2]


class AddMulRule18(RewriteRule):
    def _init_pattern_info(self) -> Tuple[OpGraph, List[OpId], List[OpId]]:
        G = OpGraph()
        x14 = G.placeholder([1])
        x11 = G.placeholder([1])
        x10 = G.placeholder([1])
        y0 = G.mul(x10, x11)
        y1 = G.add(x10, y0)
        y2 = G.mul(x14, y1)
        return G, [x14, x11, x10], [y2]

    def _create_new_subgraph(
        self, inp_ops: List[Operator], pms: MatchResult
    ) -> Tuple[OpGraph, List[OpId], List[OpId]]:
        G = OpGraph()
        x14, x11, x10 = G.add_nodes(inp_ops)
        y0 = G.mul(x11, x14)
        y1 = G.add(x14, y0)
        y2 = G.mul(x10, y1)
        return G, [x14, x11, x10], [y2]


class AddMulRule19(RewriteRule):
    def _init_pattern_info(self) -> Tuple[OpGraph, List[OpId], List[OpId]]:
        G = OpGraph()
        x13 = G.placeholder([1])
        x11 = G.placeholder([1])
        x10 = G.placeholder([1])
        y0 = G.add(x10, x11)
        y1 = G.mul(x13, y0)
        return G, [x13, x11, x10], [y1]

    def _create_new_subgraph(
        self, inp_ops: List[Operator], pms: MatchResult
    ) -> Tuple[OpGraph, List[OpId], List[OpId]]:
        G = OpGraph()
        x13, x11, x10 = G.add_nodes(inp_ops)
        y0 = G.mul(x11, x13)
        y1 = G.mul(x10, x13)
        y2 = G.add(y0, y1)
        return G, [x13, x11, x10], [y2]


class AddMulRule20(RewriteRule):
    def _init_pattern_info(self) -> Tuple[OpGraph, List[OpId], List[OpId]]:
        G = OpGraph()
        x13 = G.placeholder([1])
        x11 = G.placeholder([1])
        x10 = G.placeholder([1])
        y0 = G.mul(x10, x11)
        y1 = G.mul(x13, y0)
        y2 = G.mul(x13, y1)
        return G, [x13, x11, x10], [y2]

    def _create_new_subgraph(
        self, inp_ops: List[Operator], pms: MatchResult
    ) -> Tuple[OpGraph, List[OpId], List[OpId]]:
        G = OpGraph()
        x13, x11, x10 = G.add_nodes(inp_ops)
        y0 = G.mul(x11, x13)
        y1 = G.mul(x10, x13)
        y2 = G.mul(y0, y1)
        return G, [x13, x11, x10], [y2]


class AddMulRule21(RewriteRule):
    def _init_pattern_info(self) -> Tuple[OpGraph, List[OpId], List[OpId]]:
        G = OpGraph()
        x13 = G.placeholder([1])
        x11 = G.placeholder([1])
        x10 = G.placeholder([1])
        y0 = G.mul(x10, x11)
        y1 = G.mul(x13, y0)
        return G, [x13, x11, x10], [y1]

    def _create_new_subgraph(
        self, inp_ops: List[Operator], pms: MatchResult
    ) -> Tuple[OpGraph, List[OpId], List[OpId]]:
        G = OpGraph()
        x13, x11, x10 = G.add_nodes(inp_ops)
        y0 = G.mul(x11, x13)
        y1 = G.mul(x10, y0)
        return G, [x13, x11, x10], [y1]


class AddMulRule22(RewriteRule):
    def _init_pattern_info(self) -> Tuple[OpGraph, List[OpId], List[OpId]]:
        G = OpGraph()
        x14 = G.placeholder([1])
        x11 = G.placeholder([1])
        x10 = G.placeholder([1])
        y0 = G.mul(x10, x11)
        y1 = G.add(x11, y0)
        y2 = G.mul(x14, y1)
        return G, [x14, x11, x10], [y2]

    def _create_new_subgraph(
        self, inp_ops: List[Operator], pms: MatchResult
    ) -> Tuple[OpGraph, List[OpId], List[OpId]]:
        G = OpGraph()
        x14, x11, x10 = G.add_nodes(inp_ops)
        y0 = G.mul(x11, x14)
        y1 = G.mul(x10, y0)
        y2 = G.add(y0, y1)
        return G, [x14, x11, x10], [y2]


class AddMulRule23(RewriteRule):
    def _init_pattern_info(self) -> Tuple[OpGraph, List[OpId], List[OpId]]:
        G = OpGraph()
        x13 = G.placeholder([1])
        x11 = G.placeholder([1])
        x10 = G.placeholder([1])
        y0 = G.mul(x10, x11)
        y1 = G.mul(x10, x13)
        y2 = G.mul(y0, y1)
        return G, [x13, x11, x10], [y2]

    def _create_new_subgraph(
        self, inp_ops: List[Operator], pms: MatchResult
    ) -> Tuple[OpGraph, List[OpId], List[OpId]]:
        G = OpGraph()
        x13, x11, x10 = G.add_nodes(inp_ops)
        y0 = G.mul(x11, x13)
        y1 = G.mul(x10, y0)
        y2 = G.mul(x10, y1)
        return G, [x13, x11, x10], [y2]


class AddMulRule24(RewriteRule):
    def _init_pattern_info(self) -> Tuple[OpGraph, List[OpId], List[OpId]]:
        G = OpGraph()
        x13 = G.placeholder([1])
        x11 = G.placeholder([1])
        x10 = G.placeholder([1])
        y0 = G.mul(x10, x11)
        y1 = G.mul(x13, y0)
        y2 = G.mul(x13, y1)
        return G, [x13, x11, x10], [y2]

    def _create_new_subgraph(
        self, inp_ops: List[Operator], pms: MatchResult
    ) -> Tuple[OpGraph, List[OpId], List[OpId]]:
        G = OpGraph()
        x13, x11, x10 = G.add_nodes(inp_ops)
        y0 = G.mul(x11, x13)
        y1 = G.mul(x13, y0)
        y2 = G.mul(x10, y1)
        return G, [x13, x11, x10], [y2]


class AddMulRule25(RewriteRule):
    def _init_pattern_info(self) -> Tuple[OpGraph, List[OpId], List[OpId]]:
        G = OpGraph()
        x13 = G.placeholder([1])
        x11 = G.placeholder([1])
        x10 = G.placeholder([1])
        y0 = G.add(x10, x11)
        y1 = G.mul(x13, y0)
        return G, [x13, x11, x10], [y1]

    def _create_new_subgraph(
        self, inp_ops: List[Operator], pms: MatchResult
    ) -> Tuple[OpGraph, List[OpId], List[OpId]]:
        G = OpGraph()
        x13, x11, x10 = G.add_nodes(inp_ops)
        y0 = G.mul(x13, x11)
        y1 = G.mul(x13, x10)
        y2 = G.add(y0, y1)
        return G, [x13, x11, x10], [y2]


class AddMulRule26(RewriteRule):
    def _init_pattern_info(self) -> Tuple[OpGraph, List[OpId], List[OpId]]:
        G = OpGraph()
        x13 = G.placeholder([1])
        x11 = G.placeholder([1])
        x10 = G.placeholder([1])
        y0 = G.add(x10, x11)
        y1 = G.mul(x13, y0)
        return G, [x13, x11, x10], [y1]

    def _create_new_subgraph(
        self, inp_ops: List[Operator], pms: MatchResult
    ) -> Tuple[OpGraph, List[OpId], List[OpId]]:
        G = OpGraph()
        x13, x11, x10 = G.add_nodes(inp_ops)
        y0 = G.mul(x13, x11)
        y1 = G.mul(x10, x13)
        y2 = G.add(y0, y1)
        return G, [x13, x11, x10], [y2]
