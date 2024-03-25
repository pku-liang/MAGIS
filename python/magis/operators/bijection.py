from .base import Operator, ComputeOp
from .. import utils
from typing import List, Tuple


class EwiseOp(ComputeOp):
    def __init__(self, tag, inps: List[Operator], **attrs) -> None:
        assert len(inps) > 0
        out_shape = tuple(inps[0].out_shape)
        assert all(tuple(i.out_shape) == out_shape for i in inps[1:]), [i.out_shape for i in inps]
        super().__init__(f"ewise.{tag}", out_shape, inps, **attrs)

        self._dim_links = [[(i, i) for i in range(len(out_shape))] for _ in inps]


class EwiseBinOp(EwiseOp):
    def __init__(self, tag, a, b, commutative=False, **attrs) -> None:
        super().__init__(f"bin.{tag}", [a, b], commutative=commutative, **attrs)

    def backward(
        self, out_grad: "Operator", inps: List["Operator"]
    ) -> Tuple[List["Operator"], List[Tuple["Operator", List["Operator"]]]]:
        a, b = inps
        tag = self.tag.split(".")[-1]
        if tag == "add":
            return [out_grad, out_grad], []
        elif tag == "sub":
            bg = EwiseUniOp("neg", out_grad)
            return [out_grad, bg], [(bg, [out_grad])]
        elif tag == "mul":
            a_grad = EwiseBinOp("mul", out_grad, b)
            b_grad = EwiseBinOp("mul", out_grad, a)
            return [a_grad, b_grad], [(a_grad, [out_grad, b]), (b_grad, [out_grad, a])]
        else:
            raise NotImplementedError()


class EwiseUniOp(EwiseOp):
    def __init__(self, tag, x, **attrs) -> None:
        super().__init__(f"uni.{tag}", [x], **attrs)

    def backward(
        self, out_grad: "Operator", inps: List["Operator"]
    ) -> Tuple[List["Operator"], List[Tuple["Operator", List["Operator"]]]]:
        (x,) = inps
        tag = self.tag.split(".")[-1]
        if tag == "relu":
            mask = EwiseUniConstOp("gt", self, 0)
            xg = EwiseBinOp("mul", out_grad, mask)
            return [xg], [(mask, [self]), (xg, [out_grad, mask])]
        elif tag in ("tanh", "sigmoid"):
            xg = EwiseBinOp(f"{tag}_backward", out_grad, self)
            return [xg], [(xg, [out_grad, self])]
        elif tag == "neg":
            xg = EwiseUniOp("neg", out_grad)
            return [xg], [(xg, [out_grad])]
        elif tag == "reciprocal":
            xg1 = EwiseUniConstOp("pow", self, 2)
            xg2 = EwiseBinOp("mul", out_grad, xg1)
            xg3 = EwiseUniOp("neg", xg2)
            return [xg3], [(xg1, [self]), (xg2, [out_grad, xg1]), (xg3, [xg2])]
        elif tag == "log":
            xg = EwiseBinOp("div", out_grad, x)
            return [xg], [(xg, [out_grad, x])]
        elif tag == "exp":
            xg = EwiseBinOp("mul", out_grad, self)
            return [xg], [(xg, [out_grad, self])]
        else:
            raise NotImplementedError()


class EwiseUniConstOp(EwiseUniOp):
    def __init__(self, tag, x, const, **attrs) -> None:
        super().__init__(f"const.{tag}", x, const=const, **attrs)

    def backward(
        self, out_grad: "Operator", inps: List["Operator"]
    ) -> Tuple[List["Operator"], List[Tuple["Operator", List["Operator"]]]]:
        (x,) = inps
        const = self.attrs["const"]
        tag = self.tag.split(".")[-1]
        if tag in ("add", "sub"):
            return [out_grad], []
        elif tag in ("mul", "div"):
            xg = EwiseUniConstOp(tag, out_grad, const)
            return [xg], [(xg, [out_grad])]
        elif tag == "pow":
            if const == 2:
                mul1 = EwiseBinOp("mul", out_grad, x)
                mul2 = EwiseUniConstOp("mul", mul1, 2)
                return [mul2], [(mul1, [out_grad, x]), (mul2, [mul1])]
            else:
                pow_grad = EwiseUniConstOp("pow", x, const - 1)
                mul1 = EwiseBinOp("mul", out_grad, pow_grad)
                mul2 = EwiseUniConstOp("mul", mul1, const)
                return [mul2], [
                    (pow_grad, [x]),
                    (mul1, [out_grad, pow_grad]),
                    (mul2, [mul1]),
                ]
        else:
            raise NotImplementedError()


class ReshapeOp(ComputeOp):
    REUSE_PROF_CODE = False

    def __init__(self, inp: Operator, shape: List[int], **attrs) -> None:
        inp_shape = inp.out_shape
        assert utils.prod(inp_shape) == utils.prod(shape)
        i_idx = o_idx = 0
        i_acc = o_acc = 1
        i_last, o_last = -1, -2
        aligned_dims = []
        while i_idx < len(inp_shape) or o_idx < len(shape):
            if i_acc <= o_acc:
                if inp_shape[i_idx] > 1:
                    i_last = inp_shape[i_idx]
                    i_acc *= i_last
                i_idx += 1
            else:
                if shape[o_idx] > 1:
                    o_last = shape[o_idx]
                    o_acc *= o_last
                o_idx += 1
            if i_acc == o_acc:
                if i_last == o_last:
                    aligned_dims.append((i_idx - 1, o_idx - 1))

        super().__init__(
            "reshape",
            shape,
            [inp],
            aligned_dims=aligned_dims,
            **attrs,
        )

        self._dim_links = [aligned_dims]

    def backward(
        self, out_grad: "Operator", inps: List["Operator"]
    ) -> Tuple[List["Operator"], List[Tuple["Operator", List["Operator"]]]]:
        shape = self.inp_shapes[0]
        xg = ReshapeOp(out_grad, shape)
        return [xg], [(xg, [out_grad])]


class PermuteOp(ComputeOp):
    def __init__(self, inp: Operator, dims: List[int], **attrs) -> None:
        dims = [utils.pos_dim(d, len(inp.out_shape)) for d in dims]
        assert tuple(sorted(dims)) == tuple(range(len(dims)))
        out_shape = inp.out_shape
        out_shape = [out_shape[d] for d in dims]
        super().__init__("permute", out_shape, [inp], dims=tuple(dims), **attrs)

        self._dim_links = [[(d, i) for i, d in enumerate(dims)]]

    def backward(
        self, out_grad: "Operator", inps: List["Operator"]
    ) -> Tuple[List["Operator"], List[Tuple["Operator", List["Operator"]]]]:
        idims = sorted([(i, o) for o, i in enumerate(self.attrs["dims"])])
        idims = [o for _, o in idims]
        xg = PermuteOp(out_grad, idims)
        return [xg], [(xg, [out_grad])]
