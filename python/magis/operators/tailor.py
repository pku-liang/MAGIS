from .base import Operator, ComputeOp
from .. import utils
from typing import List, Set, Tuple


class ConcatOp(ComputeOp):
    def __init__(self, inps: List[Operator], dim) -> None:
        assert len(inps) > 0
        inp_shapes = [op.out_shape for op in inps]
        out_shape = list(inp_shapes[0])
        dim = utils.pos_dim(dim, len(out_shape))
        for shape in inp_shapes[1:]:
            x = out_shape[dim]
            out_shape[dim] = shape[dim]
            assert tuple(out_shape) == tuple(shape)
            out_shape[dim] += x
        super().__init__("concat", out_shape, inps, dim=dim)

        self._dim_links = [
            [(i, i) for i in range(len(out_shape)) if i != dim] for _ in inps
        ]

    def backward(
        self, out_grad: "Operator", inps: List["Operator"]
    ) -> Tuple[List["Operator"], List[Tuple["Operator", List["Operator"]]]]:
        dim = self.attrs["dim"]
        cur_start = 0
        xgrads = []
        links = []
        for i, shp in enumerate(self.inp_shapes):
            cur_stop = cur_start + shp[dim]
            xg = SliceOp(out_grad, dim, cur_start, cur_stop)
            xgrads.append(xg)
            links.append((xg, [out_grad]))
            cur_start = cur_stop
        return xgrads, links

    def _get_latency(self):
        return self.ZERO_LATENCY


class SliceOp(ComputeOp):
    def __init__(self, inp: Operator, dims, starts, ends) -> None:
        dims, starts, ends = (utils.to_tuple(x) for x in (dims, starts, ends))
        inp_shape = inp.out_shape
        out_shape = list(inp_shape)
        dims = tuple(utils.pos_dim(d, len(out_shape)) for d in dims)
        assert len(dims) == len(set(dims)) == len(starts) == len(ends) > 0
        for dim, start, end in zip(dims, starts, ends):
            length = inp_shape[dim]
            if end == -1:
                end = length
            assert 0 <= start < end <= length
            out_shape[dim] = end - start
        super().__init__("slice", out_shape, [inp], dims=dims, starts=starts, ends=ends)

        self._dim_links = [[(i, i) for i in range(len(out_shape)) if i not in dims]]

    def _get_latency(self):
        return self.ZERO_LATENCY

    def infer_input_linked_dims(self, out_dims: Set[int]) -> List[Set[int]]:
        ret = super().infer_input_linked_dims(out_dims)
        for d in out_dims:
            if d < 0:
                for ds in ret:
                    ds.add(d)
        return ret


class PaddingOp(ComputeOp):
    def __init__(self, inp: Operator, pads: List[Tuple[int, int]]) -> None:
        inp_shape = inp.out_shape
        out_shape = list(inp_shape)
        assert len(pads) == len(out_shape)
        pads = tuple((p, p) if isinstance(p, int) else tuple(p) for p in pads)
        for i, p in enumerate(pads):
            assert len(p) == 2
            out_shape[i] += p[0] + p[1]
        super().__init__("padding", out_shape, [inp], pads=pads)

        self._dim_links = [[(i, i) for i in range(len(out_shape)) if sum(p[i]) == 0]]

    def backward(
        self, out_grad: "Operator", inps: List["Operator"]
    ) -> Tuple[List["Operator"], List[Tuple["Operator", List["Operator"]]]]:
        dims, starts, ends = [], [], []
        for i, (p, s) in enumerate(zip(self.attrs["pads"], self.out_shape)):
            if sum(p) == 0:
                continue
            dims.append(i)
            starts.append(p[0])
            ends.append(s - p[1])
        if len(dims) == 0:
            return [out_grad], []
        xg = SliceOp(out_grad, dims, starts, ends)
        return [xg], [(xg, [out_grad])]
