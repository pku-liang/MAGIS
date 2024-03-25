from typing import List, Set

from . import ComputeOp
from .. import op_graph
from .. import utils


class FissionOp(ComputeOp):
    def __init__(
        self,
        G,
        G_fs_inp_ids,
        G_fs_out_ids,
        F,
        F_inp_ids,
        F_out_ids,
        inp_fiss_dims,
        out_fiss_dims,
        fs_length,
        fs_factor,
        **attrs,
    ):
        super().__init__(
            "fission",
            (0,),
            [G[i] for i in G_fs_inp_ids],
            **attrs,
        )
        G: op_graph.OpGraph = G
        self._n_inp_ids = len(G_fs_inp_ids)
        self.real_out_shapes = [G[i].out_shape for i in G_fs_out_ids]
        self.fs_graph: op_graph.OpGraph = F
        self.inp_fs_dims: List[Set[int]] = list(inp_fiss_dims)
        self.out_fs_dims: List[Set[int]] = list(out_fiss_dims)
        self.fs_length = fs_length
        self.fs_factor = fs_factor
        self.fs_graph._fs_inp_ids = list(F_inp_ids)
        self.fs_graph._fs_out_ids = list(F_out_ids)

        self._dim_links = [[(d, 0) for d in fs_dims] for fs_dims in self.inp_fs_dims]

    @property
    def fs_inp_ids(self):
        return self.fs_graph._fs_inp_ids

    @property
    def fs_out_ids(self):
        return self.fs_graph._fs_out_ids

    def _get_latency(self):
        return sum(self.fs_graph[i].latency for i in self.fs_graph.all_ids()) * (
            self.fs_length // self.fs_factor
        )

    def _get_out_memory(self):
        return sum(utils.prod(s) for s in self.real_out_shapes)

    def _get_digest(self):
        return utils.digest(str(self.key).encode() + self.fs_graph.digest)

    def dump(self, op_id, pre_ids, indent=0):
        base = super().dump(op_id, pre_ids, indent)
        subg = self.fs_graph.dump(indent + 1)
        return (base + "\n" + subg).strip("\n")


class FissionOutOp(ComputeOp):
    def __init__(self, inp: FissionOp, index, dims, **attrs) -> None:
        assert inp.is_fission()
        super().__init__(
            "fission_out",
            inp.real_out_shapes[index],
            [inp],
            index=index,
            dims=dims,
            **attrs,
        )
        self._dim_links = [[(0, d) for d in dims]]

    def _get_latency(self):
        return self.ZERO_LATENCY
