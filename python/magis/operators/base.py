from typing import List, Tuple, Set, Dict
from copy import copy as _shallow_copy

from .. import utils


class Operator:
    DEFAULT_IMAGE_LAYOUT = "nchw"
    # whether the instances with different shapes can reuse the same profiling code
    REUSE_PROF_CODE = True
    ZERO_LATENCY = 1e-8

    def __init__(self, tag, out_shape, inp_shapes, in_mem, **attrs) -> None:
        self.tag: str = tag
        self.out_shape: List[int] = tuple(out_shape)
        self.inp_shapes: List[List[int]] = [tuple(s) for s in inp_shapes]
        self.in_mem = in_mem  # whether the output is in memory
        self.attrs = attrs

        """表示算子的 output 和各个 input 之间的 dimension 连接
        比如 [[(a1, c1), (a2, -1)], [(b1, c2), (b2, -1)]] 表示：
        1. A 的 a1 和 C 的 c1 为 spatial-axis, a2 为 第1个 reduce-axis
        2. B 的 b1 和 C 的 c2 为 spatial-axis, b2 为 第1个 reduce-axis
        """
        self._dim_links: List[List[Tuple[int, int]]] = []
        self._loop_dims: Set[int] = None

        # may be changed with different shapes
        self._out_memory: int = -1
        self._extra_memory: int = -1
        self._latency: float = -1
        self._key = None
        self._digest = None
        self._loop_lens: Dict[int, int] = None

    def copy(self):
        new_op = _shallow_copy(self)
        new_op._out_memory = -1
        new_op._extra_memory = -1
        new_op._latency = -1
        new_op._key = None
        new_op._digest = None
        new_op._loop_lens = None
        new_op.attrs = dict(new_op.attrs)
        return new_op

    def may_copy(self, inplace=False):
        self = self if inplace else self.copy()
        return self

    def __getitem__(self, k):
        return self.attrs[k]

    def __setitem__(self, k, v):
        self.attrs[k] = v

    @property
    def latency(self):
        if self._latency < 0:
            self._latency = self._get_latency()
        return self._latency

    def _get_latency(self):
        from ..backend import BaseBackend

        if BaseBackend.CURRENT_BACKEND is not None:
            return BaseBackend.CURRENT_BACKEND.measure_operator_latency(
                self,
                BaseBackend.DEFAULT_MEASURE_NUMBER,
                BaseBackend.DEFAULT_MEASURE_REPEAT,
            )
        return self.ZERO_LATENCY

    @property
    def out_memory(self):
        if self._out_memory < 0:
            self._out_memory = self._get_out_memory()
        return self._out_memory

    def _get_out_memory(self):
        return utils.prod(self.out_shape)

    @property
    def extra_memory(self):
        if self._extra_memory < 0:
            self._extra_memory = self._get_extra_memory()
        return self._extra_memory

    def _get_extra_memory(self):
        return 0

    @property
    def key(self):
        if self._key is None:
            self._key = self._get_key()
        return self._key

    def _get_key(self):
        return (
            self.tag + str(sorted(self.attrs.items())),
            str(self.out_shape) + str(self.inp_shapes),
        )

    @property
    def digest(self):
        if self._digest is None:
            self._digest = self._get_digest()
        return self._digest

    def _get_digest(self):
        return utils.digest(str(self.key).encode())

    def _merge_args_digest(self, args_digest):
        return utils.digest(b"".join([self.digest, *args_digest]))

    def dump(self, op_id, pre_ids, indent=0):
        return "{spaces}${dst} {{shape={shape}}} := {op}({srcs}) {attrs}".format(
            spaces="  " * indent,
            dst=op_id,
            shape=list(self.out_shape),
            op=self.tag,
            srcs=", ".join("$" + str(pi) for pi in pre_ids),
            attrs=self.attrs,
        )

    def is_weight(self):
        return self.attrs.get("is_weight", False)

    def is_mem_op(self):
        return self.tag in ("store", "load")

    def is_load(self):
        return self.tag == "load"

    def is_store(self):
        return self.tag == "store"

    def is_fission(self):
        return self.tag == "fission"

    def is_remat(self):
        # return self.attrs.get("is_remat", False)
        return getattr(self, "_is_remat", False)

    def is_placeholder(self):
        return self.tag == "placeholder"

    def multi_out_shapes(self):
        return self.attrs.get("multi_out_shapes", None)

    def backward(
        self, out_grad: "Operator", inps: List["Operator"]
    ) -> Tuple[List["Operator"], List[Tuple["Operator", List["Operator"]]]]:
        """对算子进行反向求导

        Parameters
        ----------
        out_grad: Operator
            算子的输出的梯度

        inps: List[Operator]
            算子的输入算子

        Returns
        -------
        inp_grads: List[Operator]
            算子的各个输入算子的梯度

        links: List[Tuple[Operator, List[Operator]]]
            反向求导过程中产生的各个新算子以及其输入算子
            [(c, [a, b]), (d, [a, c])] 意思是 c = f1(a, b); d = f2(a, c)
        """
        if len(inps) == 0:
            return [], []
        raise NotImplementedError(f"{type(self)} currently does not support backward")

    def infer_output_linked_dims(self, inp_dims: List[Set[int]]) -> Set[int]:
        """给定各个输入算子的一些维度, 推导出该算子和那些维度相连的维度

        Parameters
        ----------
        inp_dims: List[List[int]]
            各个输入算子的给定维度

        Returns
        -------
        out_dims: Set[int]
            输出算子的相连维度
        """
        ret = set()
        for ds, ls in zip(inp_dims, self._dim_links):
            ret.update(d1 for (d0, d1) in ls if d0 in ds)
        return ret

    def infer_input_linked_dims(self, out_dims: Set[int]) -> List[Set[int]]:
        """给定各个输出算子的一些维度, 推导出各个输入算子的相连维度

        Parameters
        ----------
        out_dims: List[int]
            输出算子的一些维度

        Returns
        -------
        inp_dims: List[Set[int]]
            各个输入算子的相连维度
        """
        ret = []
        for ls in self._dim_links:
            ret.append({d0 for (d0, d1) in ls if d1 in out_dims})
        return ret

    @property
    def loop_dims(self):
        if self._loop_dims is None:
            self._loop_dims = self._get_loop_dims()
        return self._loop_dims

    def _get_loop_dims(self):
        return {d1 for links in self._dim_links for (_, d1) in links}

    @property
    def loop_lens(self):
        if self._loop_lens is None:
            self._loop_lens = self._get_loop_lens()
        return self._loop_lens

    def _get_loop_lens(self):
        ret = dict()
        for shape, links in zip(self.inp_shapes, self._dim_links):
            for d0, d1 in links:
                d1_len = ret.get(d1, None)
                if d1 < 0:
                    d0_len = shape[d0]
                    if d1_len is None:
                        ret[d1] = d0_len
                    else:
                        assert d1_len == d0_len
                else:
                    if d1_len is None:
                        ret[d1] = self.out_shape[d1]
        return ret


class PlaceholderOp(Operator):
    _USED_NAMES = set()

    def __init__(self, shape, name_hint="p", in_mem=True, is_weight=False) -> None:
        suffix = len(self._USED_NAMES)
        name = name_hint
        while name in self._USED_NAMES:
            name = name_hint + str(suffix)
            suffix += 1
        self._USED_NAMES.add(name)
        super().__init__(
            "placeholder", shape, [], in_mem, name=name, is_weight=is_weight
        )

    def _get_latency(self):
        return self.ZERO_LATENCY

    def _get_key(self):
        return (
            self.tag
            + str(
                sorted(dict(in_mem=self.in_mem, is_weight=self["is_weight"]).items())
            ),
            str(self.out_shape) + str(self.inp_shapes),
        )


class ComputeOp(Operator):
    def __init__(self, tag, out_shape, inps: List[Operator], **attrs) -> None:
        assert all(op.in_mem for op in inps)
        super().__init__(tag, out_shape, [op.out_shape for op in inps], True, **attrs)


class StoreOp(Operator):
    def __init__(self, inp: Operator, **attrs) -> None:
        assert inp.in_mem
        super().__init__("store", inp.out_shape, [inp.out_shape], False, **attrs)

        self._dim_links = [[(i, i) for i in range(len(inp.out_shape))]]


class LoadOp(Operator):
    def __init__(self, inp: Operator, **attrs) -> None:
        assert not inp.in_mem
        super().__init__("load", inp.out_shape, [inp.out_shape], True, **attrs)

        self._dim_links = [[(i, i) for i in range(len(inp.out_shape))]]


class IndexOutOp(ComputeOp):
    def __init__(self, inp: Operator, index, dim_links, **attrs) -> None:
        out_shapes = inp.multi_out_shapes()
        assert out_shapes is not None
        super().__init__(
            "index_out",
            out_shapes[index],
            [inp],
            index=index,
            dim_links=dim_links,
            **attrs,
        )
        self._dim_links = [list(dim_links)]

    def backward(
        self, out_grad: Operator, inps: List[Operator]
    ) -> Tuple[List[Operator], List[Tuple[Operator, List[Operator]]]]:
        out_grad.attrs["grad_index"] = self.attrs["index"]
        return [out_grad], []

    def _get_latency(self):
        return self.ZERO_LATENCY
