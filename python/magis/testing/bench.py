import time
import os

import magis.scheduler as SCH
import magis.optimizer as OPT
import magis.backend as BAK
from magis.op_graph import OpGraph, OpId
from magis.utils import LOG
from magis import utils
from .configs import get_configured_mutator


def setup_training_graph(G: OpGraph, y: OpId, inplace=True, update_weight=True):
    G = G.may_copy(inplace=inplace)
    y_bar = G.placeholder(G[y].out_shape, "y_bar")
    loss = G.pow_const(G.sub(y, y_bar), 2)
    return G.backward(loss, inplace=True, update_weight=update_weight)


def run_optimization(
    G: OpGraph,
    name,
    opt: OPT.BaseOptimizer = None,
    scheduler=None,
    mutator=None,
    dtype="float32",
    dump_file=None,
    save_graph=True,
    save_records=True,
    **kwargs,
):
    wgt_mem = sum(G[v].out_memory for v in G.all_ids() if G[v].is_weight())
    with BAK.TorchCudaBackend(dtype=dtype) as bknd:
        if not isinstance(opt, OPT.BaseOptimizer):
            scheduler = scheduler or SCH.RefineMemOpRpoScheduler(adjust_load_op=False)
            mutator = mutator or get_configured_mutator(**kwargs)
            opt = (opt or OPT.RelaxOptimizer)(scheduler, bknd, mutator, **kwargs)
        G, sched = opt.run(G)
        sim_res = opt._best_state.sim_res
        opt._always_simulation = False
        run_res = opt._profile(state=opt._best_state, number=4, repeat=5)
        opt._init_state.run_res = opt._profile(
            state=opt._init_state, number=4, repeat=5
        )

        ll = opt._lat_limit
        ml = opt._mem_limit
        dml = opt._dev_mem_limit
        llr = opt._lat_limit_ratio
        mlr = opt._mem_limit_ratio
        if save_graph:
            os.makedirs("./data", exist_ok=True)
            utils.save_pickle(os.path.join("./data", name + ".pkl"), (G, sched))
        if save_records:
            with open(os.path.join("./data", name + ".tmp.csv"), "w") as fp:
                for rec in opt._records:
                    print(*rec, sep=",", file=fp)
        res = (
            name,
            *(dml, ll, ml, llr, mlr),
            wgt_mem,
            *run_res,
            *(sim_res.latency, sim_res.peak_memory),
            *opt._init_state.run_res,
            *(opt._init_state.sim_res.latency, opt._init_state.sim_res.peak_memory),
        )
        LOG.info(f"run_results:{res}")
        if dump_file:
            print(*res, file=dump_file, flush=True, sep=",")
