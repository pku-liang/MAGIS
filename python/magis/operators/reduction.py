from .base import Operator, ComputeOp
from .. import utils
from typing import List, Tuple


class ReduceOp(ComputeOp):
    def __init__(self, tag, inp: Operator, dim, keepdim=False) -> None:
        out_shape = list(inp.out_shape)
        dim = utils.pos_dim(dim, len(out_shape))
        out_shape[dim] = 1
        if not keepdim:
            out_shape.pop(dim)
        super().__init__(f"reduce.{tag}", out_shape, [inp], dim=dim, keepdim=keepdim)

        self._dim_links = [
            [
                (i, i - 1 if (not keepdim and i > dim) else i) if i != dim else (i, -1)
                for i in range(len(out_shape))
            ]
        ]


class SoftmaxOp(ComputeOp):
    def __init__(self, inp: Operator, dim) -> None:
        out_shape = inp.out_shape
        dim = utils.pos_dim(dim, len(out_shape))
        out_shape[dim]
        super().__init__(f"softmax", out_shape, [inp], dim=dim)

        self._dim_links = [[(i, i) for i in range(len(out_shape)) if i != dim]]

    def backward(
        self, out_grad: "Operator", inps: List["Operator"]
    ) -> Tuple[List["Operator"], List[Tuple["Operator", List["Operator"]]]]:
        xg = SoftmaxBwdOp(out_grad, self, self.attrs["dim"])
        return [xg], [(xg, [out_grad, self])]


# ugly
class SoftmaxBwdOp(ComputeOp):
    def __init__(self, out_grad: Operator, out: Operator, dim: int) -> None:
        dim = utils.pos_dim(dim, len(out.out_shape))
        super().__init__("softmax_bwd", out.out_shape, [out_grad, out], dim=dim)

        self._dim_links = [[(i, i) for i in range(len(out.out_shape)) if i != dim]]


class LayerNormOp(ComputeOp):
    def __init__(self, inp: Operator, norm_shape, **attrs) -> None:
        super().__init__("layer_norm", inp.out_shape, [inp], **attrs)
        norm_shape = tuple(norm_shape)
        assert norm_shape == tuple(inp.out_shape[-len(norm_shape) :])
        mean_shape = list(inp.out_shape)
        for i in range(len(norm_shape)):
            mean_shape[-(i + 1)] = 1
        mean_shape = tuple(mean_shape)
        self.attrs["multi_out_shapes"] = (inp.out_shape, mean_shape, mean_shape)
        self.attrs["norm_shape"] = norm_shape

        self._dim_links = [
            [(i, i) for i in range(len(inp.out_shape) - len(norm_shape))]
        ]

    def _get_out_memory(self):
        return sum(utils.prod(s) for s in self.attrs["multi_out_shapes"])

    def backward(
        self, out_grad: List[Operator], inps: List[Operator], outs: List[Operator]
    ) -> Tuple[List[Operator], List[Tuple[Operator, List[Operator]]]]:
        _, mean, std = outs
        xg = LayerNormBwdOp(out_grad, inps[0], mean, std, self.attrs["norm_shape"])
        return [xg], [(xg, [out_grad, inps[0], mean, std])]


# ugly
class LayerNormBwdOp(ComputeOp):
    def __init__(
        self,
        out_grad: Operator,
        inp: Operator,
        mean: Operator,
        std: Operator,
        norm_shape,
        **attrs,
    ) -> None:
        super().__init__(
            "layer_norm_bwd",
            inp.out_shape,
            [out_grad, inp, mean, std],
            norm_shape=norm_shape,
            **attrs,
        )

        self._dim_links = [inp._dim_links[0] for _ in self.inp_shapes]
