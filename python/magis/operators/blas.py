from .base import Operator, ComputeOp
from typing import List, Tuple


class MatmulOp(ComputeOp):
    """表示的是一般的 Matmul 和 Batch-Matmul 算子
    [*b, m, k] x [*b, k, n] -> [*b, m, n]
    """

    def __init__(
        self,
        a: Operator,
        b: Operator,
        trans_a=False,
        trans_b=False,
        trans_c=False,
    ) -> None:
        if trans_a:
            *ba, k, m = a.out_shape
        else:
            *ba, m, k = a.out_shape
        if trans_b:
            *bb, n, k1 = b.out_shape
        else:
            *bb, k1, n = b.out_shape
        assert k == k1
        assert tuple(ba) == tuple(bb)
        c_shape = (*ba, n, m) if trans_c else (*ba, m, n)

        super().__init__(
            "matmul",
            c_shape,
            [a, b],
            trans_a=trans_a,
            trans_b=trans_b,
            trans_c=trans_c,
        )

        nb = len(ba)
        self._dim_links = [
            [
                *((i, i) for i in range(nb)),
                (nb + int(trans_a), nb + int(trans_c)),
                (nb + int(not trans_a), -1),
            ],
            [
                *((i, i) for i in range(nb)),
                (nb + int(not trans_b), nb + int(not trans_c)),
                (nb + int(trans_b), -1),
            ],
        ]

    def backward(
        self, out_grad: "Operator", inps: List["Operator"]
    ) -> Tuple[List["Operator"], List[Tuple["Operator", List["Operator"]]]]:
        a, b = inps
        ta, tb, tc = (self.attrs[f"trans_{x}"] for x in "abc")
        a_grad = MatmulOp(out_grad, b, tc, not tb, ta)
        b_grad = MatmulOp(a, out_grad, not ta, tc, tb)
        return [a_grad, b_grad], [(a_grad, [out_grad, b]), (b_grad, [a, out_grad])]


class FlexMatmulOp(ComputeOp):
    """更灵活的 Matmul / Batch-Matmul, 允许多个 spatial-axis 和 reduce-axis
    [*b, *m, *k] x [*b, *k, *n] -> [*b, *m, *n]
    """

    REUSE_PROF_CODE = False

    def __init__(
        self,
        a: Operator,
        b: Operator,
        trans_a=False,
        trans_b=False,
        trans_c=False,
        n_batch_dims=0,
        n_reduce_dims=1,
        a_pivots=None,
        b_pivots=None,
        **attrs,
    ) -> None:
        assert n_reduce_dims > 0
        assert n_batch_dims + n_reduce_dims < len(a.out_shape)
        assert n_batch_dims + n_reduce_dims < len(b.out_shape)
        bs1 = a.out_shape[:n_batch_dims]
        bs2 = b.out_shape[:n_batch_dims]
        assert tuple(bs1) == tuple(bs2)
        if trans_a:
            ks1 = a.out_shape[n_batch_dims : n_batch_dims + n_reduce_dims]
            ms = a.out_shape[n_batch_dims + n_reduce_dims :]
        else:
            ms = a.out_shape[n_batch_dims:-n_reduce_dims]
            ks1 = a.out_shape[-n_reduce_dims:]
        if trans_b:
            ns = b.out_shape[n_batch_dims:-n_reduce_dims]
            ks2 = b.out_shape[-n_reduce_dims:]
        else:
            ks2 = b.out_shape[n_batch_dims : n_batch_dims + n_reduce_dims]
            ns = b.out_shape[n_batch_dims + n_reduce_dims :]
        assert tuple(ks1) == tuple(ks2)
        out_shape = (*bs1, *ns, *ms) if trans_c else (*bs1, *ms, *ns)

        nb = n_batch_dims
        nk = n_reduce_dims
        nm = len(a.out_shape) - nb - nk
        nn = len(b.out_shape) - nb - nk

        ap2 = nb + (nk if trans_a else nm)
        bp2 = nb + (nn if trans_b else nk)
        a_pivots = a_pivots or (0, nb, ap2, len(a.out_shape))
        b_pivots = b_pivots or (0, nb, bp2, len(b.out_shape))

        am_base = nb + nk * trans_a
        cm_base = nb + nn * trans_c
        ak_base = nb + nm * (not trans_a)
        bn_base = nb + nk * (not trans_b)
        cn_base = nb + nm * (not trans_c)
        bk_base = nb + nn * trans_b

        super().__init__(
            "flex_matmul",
            out_shape,
            [a, b],
            trans_a=trans_a,
            trans_b=trans_b,
            trans_c=trans_c,
            n_batch_dims=n_batch_dims,
            n_reduce_dims=n_reduce_dims,
            a_pivots=a_pivots,
            b_pivots=b_pivots,
            **attrs,
        )

        self._dim_links = [
            [
                *((i, i) for i in range(nb)),
                *((am_base + i, cm_base + i) for i in range(nm)),
                *((ak_base + i, -(i + 1)) for i in range(nk)),
            ],
            [
                *((i, i) for i in range(nb)),
                *((bn_base + i, cn_base + i) for i in range(nn)),
                *((bk_base + i, -(i + 1)) for i in range(nk)),
            ],
        ]

    def backward(
        self, out_grad: "Operator", inps: List["Operator"]
    ) -> Tuple[List["Operator"], List[Tuple["Operator", List["Operator"]]]]:
        a, b = inps
        ta, tb, tc = (self.attrs[f"trans_{x}"] for x in "abc")
        nb = self.attrs["n_batch_dims"]
        nr = self.attrs["n_reduce_dims"]
        nm = len(a.out_shape) - nb - nr
        nn = len(b.out_shape) - nb - nr
        a_grad = FlexMatmulOp(out_grad, b, tc, not tb, ta, nb, nn)
        b_grad = FlexMatmulOp(a, out_grad, not ta, tc, tb, nb, nm)
        return [a_grad, b_grad], [(a_grad, [out_grad, b]), (b_grad, [a, out_grad])]
