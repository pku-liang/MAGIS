import pickle
import os
from typing import List

from ..operators import Operator, FissionOp
from ..op_graph import OpGraph


class BaseBackend:
    CURRENT_BACKEND: "BaseBackend" = None
    _SAVED_BACKENDS: List["BaseBackend"] = []
    DEFAULT_MEASURE_NUMBER = 20
    DEFAULT_MEASURE_REPEAT = 5
    _OPEN = open
    OOM_ERROR = Exception

    def __init__(self, cache_file=True) -> None:
        if cache_file:
            if not isinstance(cache_file, str):
                cache_file = os.path.join(
                    os.getenv("HOME"), ".cache/magis.backend.prof_result_cache.pkl"
                )
        else:
            cache_file = None

        self._prof_result_cache = dict()
        self._cache_file = cache_file
        self._memory_limit = None

        if cache_file is not None:
            if os.path.exists(cache_file):
                with open(cache_file, "rb") as fp:
                    try:
                        self._prof_result_cache = pickle.load(fp)
                    except EOFError:
                        pass
            else:
                os.makedirs(os.path.dirname(cache_file), exist_ok=True)

    def __del__(self):
        if self._cache_file is not None:
            with self._OPEN(self._cache_file, "wb") as fp:
                pickle.dump(self._prof_result_cache, fp)

    def _measure_operator_latency(self, op: Operator, number=4, repeat=5) -> float:
        raise NotImplementedError()

    def measure_operator_latency(self, op: Operator, number=4, repeat=5) -> float:
        assert not op.is_fission()
        if op._latency >= 0:
            return op._latency
        key = self._get_op_key(op)
        res = self._prof_result_cache.get(key, None)
        if res is not None:
            op._latency = res
            return res
        res = self._measure_operator_latency(op, number, repeat)
        self._prof_result_cache[key] = res
        op._latency = res
        return res

    def _get_op_key(self, op: Operator):
        return op.key

    def __enter__(self):
        BaseBackend._SAVED_BACKENDS.append(BaseBackend.CURRENT_BACKEND)
        BaseBackend.CURRENT_BACKEND = self
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        BaseBackend.CURRENT_BACKEND = BaseBackend._SAVED_BACKENDS.pop()

    def reset(self, graph: OpGraph = None, bwd_id=None):
        raise NotImplementedError()

    def measure_graph_latency(self, number=4, repeat=5, **kwargs) -> float:
        raise NotImplementedError()

    def measure_graph_peak_memory(self, **kwargs) -> int:
        raise NotImplementedError()

    def _get_memory_limit(self):
        return float("inf")

    @property
    def memory_limit(self):
        if self._memory_limit is None:
            self._memory_limit = self._get_memory_limit()
        return self._memory_limit


class CodeGenBackend(BaseBackend):
    def __init__(self, cache_file=True) -> None:
        super().__init__(cache_file)

    def _gen_placeholder(self, op: Operator, x, name, **_) -> str:
        raise NotImplementedError()

    def _gen_matmul(self, op: Operator, c, a, b, trans_a, trans_b, trans_c, **_) -> str:
        raise NotImplementedError()

    def _gen_flex_matmul(
        self,
        op: Operator,
        c,
        a,
        b,
        trans_a,
        trans_b,
        trans_c,
        a_pivots,
        b_pivots,
        **_,
    ) -> str:
        raise NotImplementedError()

    def _gen_conv2d(self, op: Operator, o, i, w, stride, padding, dilation, **_) -> str:
        raise NotImplementedError()

    def _gen_conv2d_bwd_inp(
        self, op: Operator, ig, og, w, stride, padding, dilation, layout, **_
    ) -> str:
        raise NotImplementedError()

    def _gen_conv2d_bwd_wgt(
        self, op: Operator, wg, i, og, stride, padding, dilation, layout, **_
    ) -> str:
        raise NotImplementedError()

    def _gen_pool2d(self, op: Operator, y, x, kernel_size, padding, stride, **_) -> str:
        raise NotImplementedError()

    def _gen_pool2d_bwd(
        self, op: Operator, ig, og, i, kernel_size, padding, stride, layout, **_
    ) -> str:
        raise NotImplementedError()

    def _gen_ewise_bin(self, op: Operator, c, a, b, commutative, **_) -> str:
        raise NotImplementedError()

    def _gen_ewise_uni(self, op: Operator, y, x, **_) -> str:
        raise NotImplementedError()

    def _gen_ewise_uni_const(self, op: Operator, y, x, const, **_) -> str:
        raise NotImplementedError()

    def _gen_reduce(self, op: Operator, y, x, dim, keepdim, **_) -> str:
        raise NotImplementedError()

    def _gen_softmax(self, op: Operator, y, x, dim, **_) -> str:
        raise NotImplementedError()

    def _gen_softmax_bwd(self, op: Operator, xg, og, o, dim, **_) -> str:
        raise NotImplementedError()

    def _gen_layer_norm(self, op: Operator, y, x, norm_shape, **_) -> str:
        raise NotImplementedError()

    def _gen_layer_norm_bwd(
        self, op: Operator, xg, og, x, mean, std, norm_shape, **_
    ) -> str:
        raise NotImplementedError()

    def _gen_slice(self, op: Operator, y, x, dim, start, end, **_) -> str:
        raise NotImplementedError()

    def _gen_concat(self, op: Operator, y, *xs, dim=None, **_) -> str:
        raise NotImplementedError()

    def _gen_padding(self, op: Operator, y, x, pads, **_) -> str:
        raise NotImplementedError()

    def _gen_interpolate(self, op: Operator, y, x, size, scale, mode, **_) -> str:
        raise NotImplementedError()

    def _gen_interpolate_bwd(
        self, op: Operator, ig, og, out_size, layout, mode, **_
    ) -> str:
        raise NotImplementedError()

    def _gen_reshape(self, op: Operator, y, x, **_) -> str:
        raise NotImplementedError()

    def _gen_permute(self, op: Operator, y, x, dims, **_) -> str:
        raise NotImplementedError()

    def _gen_load(self, op: Operator, y, x, **_) -> str:
        raise NotImplementedError()

    def _gen_store(self, op: Operator, y, x, **_) -> str:
        raise NotImplementedError()

    def _gen_fission_out(self, op: Operator, y, x, index, **_) -> str:
        raise NotImplementedError()

    def _gen_index_out(self, op: Operator, y, x, index, **_) -> str:
        raise NotImplementedError()

    def _gen_fission(self, op: FissionOp, y, *xs, **_) -> str:
        raise NotImplementedError()

    def _gen_return(self, op: Operator, y, *xs, **_) -> str:
        raise NotImplementedError()

    def gen_operator(self, op: Operator, out_str: str, inp_strs: List[str]) -> str:
        tags = op.tag.split(".")
        for i in range(len(tags), 0, -1):
            method = getattr(self, "_gen_" + "_".join(tags[:i]), None)
            if method is not None:
                return method(op, out_str, *inp_strs, **op.attrs)
        raise NotImplementedError(f"codegen for operator {op.tag} is not implemented")
