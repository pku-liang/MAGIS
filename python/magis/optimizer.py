import heapq
import time
from typing import List, Tuple
from dataclasses import dataclass
from collections import namedtuple
from contextlib import contextmanager

from magis.backend import BaseBackend
from magis.transform import GraphMutator

from .scheduler import BaseScheduler
from .op_graph import OpGraph, HierSched
from .simulator import SimResult
from .backend import BaseBackend
from .transform import GraphMutator
from .utils import LOG
from . import utils


RunResult = namedtuple("RunResult", ["real", "latency", "peak_memory"])


@dataclass
class State:
    graph: OpGraph
    sched: HierSched
    sim_res: SimResult
    run_res: RunResult = None

    history: List[str] = ()

    @property
    def lat(self):
        return (self.run_res or self.sim_res).latency

    @property
    def mem(self):
        return (self.run_res or self.sim_res).peak_memory

    @property
    def real(self):
        return self.run_res and self.run_res.real

    @contextmanager
    def unreal_context(self):
        run_res = self.run_res
        self.run_res = None
        try:
            yield self
        finally:
            self.run_res = run_res

    def __str__(self) -> str:
        return (
            f"State(run_res={tuple(self.run_res or [])}, "
            f"sim_res={(self.sim_res.latency, self.sim_res.peak_memory)}, "
            f"digest={int.from_bytes(self.graph.digest, 'big')}, "
            f"history={self.history})"
        )


class BaseOptimizer:
    def __init__(
        self,
        scheduler: BaseScheduler,
        backend: BaseBackend,
        mutator: GraphMutator,
        mem_limit=None,
        lat_limit=None,
        mem_limit_ratio=None,
        lat_limit_ratio=None,
        time_budget: int = float("inf"),
        iter_budget: int = float("inf"),
        always_simulation=False,
        number=2,
        repeat=3,
        **kwargs,
    ) -> None:
        self._scheduler = scheduler
        self._backend = backend
        self._mutator = mutator
        self._mem_limit = mem_limit
        self._lat_limit = lat_limit
        self._mem_limit_ratio = mem_limit_ratio
        self._lat_limit_ratio = lat_limit_ratio
        self._time_budget = time_budget
        self._iter_budget = iter_budget
        self._always_simulation = always_simulation
        self._dev_mem_limit = self._backend.memory_limit
        self._number = number
        self._repeat = repeat

    def _profile(
        self,
        G: OpGraph = None,
        sched: HierSched = None,
        sim_res: SimResult = None,
        state: State = None,
        number=None,
        repeat=None,
    ) -> RunResult:
        number = number or self._number
        repeat = repeat or self._repeat
        if state is not None:
            G, sched, sim_res = state.graph, state.sched, state.sim_res
        assert G is not None and sched is not None
        G.import_hier_sched(sched)
        sim_res = sim_res or self._scheduler._sim.run(G)
        if self._always_simulation:
            return RunResult(False, sim_res.latency, sim_res.peak_memory)
        if sim_res.peak_memory > self._dev_mem_limit:
            LOG.warning(
                f"exceed device memory limit ({self._dev_mem_limit} elements), "
                "use SimResult instead of RunResult"
            )
            return RunResult(False, sim_res.latency, sim_res.peak_memory)
        try:
            self._backend.reset(G)
            return RunResult(
                True,
                self._backend.measure_graph_latency(number=number, repeat=repeat),
                self._backend.measure_graph_peak_memory(),
            )
        except self._backend.OOM_ERROR:
            LOG.warning("OOM, use SimResult instead of RunResult")
            self._dev_mem_limit = min(self._dev_mem_limit, sim_res.peak_memory * 0.9)
            self._scheduler._mem_limit = min(
                self._scheduler._mem_limit or float("inf"), self._dev_mem_limit
            )
            LOG.warning(f"update device memory limit to {self._dev_mem_limit}")
            return RunResult(False, sim_res.latency, sim_res.peak_memory)
        finally:
            self._backend.reset()

    def _reset(self, G: OpGraph):
        self._init_state = State(G, *self._scheduler.run(G))
        _, lat, mem = self._init_state.run_res = self._profile(
            state=self._init_state, number=4, repeat=5
        )
        if self._mem_limit is None and self._mem_limit_ratio is not None:
            self._mem_limit = mem * self._mem_limit_ratio
        if self._lat_limit is None and self._lat_limit_ratio is not None:
            self._lat_limit = lat * self._lat_limit_ratio

        self._scheduler._mem_limit = min(
            self._mem_limit or float("inf"), self._dev_mem_limit
        )
        self._scheduler._lat_limit = self._lat_limit
        LOG.info(
            "init_search: "
            f"init_state={self._init_state}, "
            f"lat_limit_ratio={self._lat_limit_ratio}, lat_limit={self._lat_limit}, "
            f"mem_limit_ratio={self._mem_limit_ratio}, mem_limit={self._mem_limit}"
        )

    def run(self, G: OpGraph) -> Tuple[OpGraph, HierSched]:
        self._reset(G)
        return self._run(G)

    def _run(self, G: OpGraph) -> Tuple[OpGraph, HierSched]:
        raise NotImplementedError()


class RelaxOptimizer(BaseOptimizer):
    def __init__(
        self,
        scheduler: BaseScheduler,
        backend: BaseBackend,
        mutator: GraphMutator,
        mem_limit=None,
        lat_limit=None,
        mem_limit_ratio=None,
        lat_limit_ratio=None,
        time_budget: int = float("inf"),
        iter_budget: int = float("inf"),
        always_simulation=False,
        number=2,
        repeat=3,
        relax_ratio=1.1,
        early_stop=None,
        stop_when_mem_met=False,
        **kwargs,
    ) -> None:
        super().__init__(
            scheduler,
            backend,
            mutator,
            mem_limit,
            lat_limit,
            mem_limit_ratio,
            lat_limit_ratio,
            time_budget,
            iter_budget,
            always_simulation,
            number,
            repeat,
            **kwargs,
        )
        self._delta = relax_ratio
        self._early_stop = early_stop
        self._stop_when_mem_met = stop_when_mem_met

    def _reset(self, G: OpGraph):
        super()._reset(G)

        dml = self._dev_mem_limit
        ml = self._mem_limit
        ll = self._lat_limit

        if ml is not None:
            f_state_key = lambda state, d1=1, d2=1: (
                not state.real,
                -1 if state.mem * d1 <= dml * d2 else state.mem * d1,
                -1 if state.mem * d1 <= ml * d2 else state.mem * d1,
                state.lat * d1,
            )
        elif ll is not None:
            f_state_key = lambda state, d1=1, d2=1: (
                not state.real,
                -1 if state.mem * d1 <= dml * d2 else state.mem * d1,
                -1 if state.lat * d1 <= ll * d2 else state.lat * d1,
                state.mem * d1,
            )
        else:
            f_state_key = lambda state, d1=1, d2=1: (
                not state.real,
                state.mem * d1,
                state.lat * d1,
            )
        self._f_state_key = f_state_key

        class _State(State):
            def __lt__(self, other):
                return f_state_key(self) < f_state_key(other)

        self._State = _State

        self._best_state = _State(
            self._init_state.graph,
            self._init_state.sched,
            self._init_state.sim_res,
            self._init_state.run_res,
        )
        self._Q = [self._best_state]
        self._visited = {self._best_state.graph.digest}
        self._n_generated = 0
        self._records = []
        self._early_stop_cnt = 0

    def _is_tolerable(self, a: State, b: State):
        return self._f_state_key(a, d2=self._delta) <= self._f_state_key(
            b, d1=self._delta, d2=self._delta
        )

    def _push(self, G: OpGraph, history=()):
        if G is None:
            return
        self._n_generated += 1
        if G.digest in self._visited:
            return
        self._visited.add(G.digest)
        state = self._State(G, *self._scheduler.run(G), history=history)
        with self._best_state.unreal_context():
            if self._is_tolerable(state, self._best_state):
                heapq.heappush(self._Q, state)
                LOG.debug("push state: {}", state)

    def _run(self, G: OpGraph) -> Tuple[OpGraph, HierSched]:
        cur_iter = 0
        init_time = time.perf_counter()

        while len(self._Q) > 0:
            cur_state = heapq.heappop(self._Q)
            has_new_best = False
            try:
                cur_state.run_res = self._profile(state=cur_state)
            except Exception as e:
                LOG.exception(e)
                continue
            LOG.info(f"pop state: {cur_state}")
            if cur_state < self._best_state:
                self._best_state = cur_state
                has_new_best = True
                LOG.info(f"new best state: {cur_state}")
                if (
                    self._stop_when_mem_met
                    and self._best_state.mem < self._dev_mem_limit
                    and self._mem_limit is not None
                    and self._best_state.mem < self._mem_limit
                ):
                    break

            for mutation, info in self._mutator.gen_mutations(
                cur_state.graph, sim_res=cur_state.sim_res, sched=cur_state.sched
            ):
                try:
                    self._push(mutation(), cur_state.history + (info,))
                except Exception as e:
                    LOG.exception(e)
                    continue

            if not has_new_best:
                self._early_stop_cnt += 1
            else:
                self._early_stop_cnt = 0

            cur_iter += 1
            cur_time = time.perf_counter() - init_time
            self._records.append(
                (
                    cur_iter,
                    cur_time,
                    self._n_generated,
                    len(self._visited),
                    *self._best_state.run_res,
                    self._best_state.sim_res.latency,
                    self._best_state.sim_res.peak_memory,
                )
            )
            LOG.info(f"visited/generated: {len(self._visited)}/{self._n_generated}")
            LOG.info(f"current iter consumption: {cur_iter}/{self._iter_budget}")
            if cur_iter >= self._iter_budget:
                break
            LOG.info(f"current time consumption: {cur_time}/{self._time_budget}")
            if cur_time >= self._time_budget:
                break
            if (
                self._early_stop is not None
                and self._early_stop_cnt >= self._early_stop
            ):
                LOG.info(
                    f"{self._early_stop} iters have not update best state, early stop"
                )
                break

        LOG.info(f"final best state: {self._best_state}")
        return self._best_state.graph, self._best_state.sched


class RelaxOptimizerV1(BaseOptimizer):
    def __init__(
        self,
        scheduler: BaseScheduler,
        backend: BaseBackend,
        mutator: GraphMutator,
        mem_limit=None,
        lat_limit=None,
        mem_limit_ratio=None,
        lat_limit_ratio=None,
        time_budget: int = float("inf"),
        iter_budget: int = float("inf"),
        always_simulation=False,
        number=2,
        repeat=3,
        relax_ratio=1.1,
        beam_width=1,
        early_stop=10,
        **kwargs,
    ) -> None:
        super().__init__(
            scheduler,
            backend,
            mutator,
            mem_limit,
            lat_limit,
            mem_limit_ratio,
            lat_limit_ratio,
            time_budget,
            iter_budget,
            always_simulation,
            number,
            repeat,
            **kwargs,
        )
        self._delta = relax_ratio
        self._beam = beam_width
        self._early_stop = early_stop

    def _reset(self, G: OpGraph):
        super()._reset(G)

        dml = self._dev_mem_limit
        ml = self._mem_limit
        ll = self._lat_limit

        if ml is not None:
            f_state_key = lambda state, d1=1, d2=1: (
                not state.real,
                -1 if state.mem * d1 <= dml * d2 else state.mem * d1,
                -1 if state.mem * d1 <= ml * d2 else state.mem * d1,
                state.lat * d1,
            )
        elif ll is not None:
            f_state_key = lambda state, d1=1, d2=1: (
                not state.real,
                -1 if state.mem * d1 <= dml * d2 else state.mem * d1,
                -1 if state.lat * d1 <= ll * d2 else state.lat * d1,
                state.mem * d1,
            )
        else:
            f_state_key = lambda state, d1=1, d2=1: (
                not state.real,
                state.mem * d1,
                state.lat * d1,
            )
        self._f_state_key = f_state_key

        class _State(State):
            def __lt__(self, other):
                return f_state_key(self) < f_state_key(other)

        self._State = _State

        self._best_state = _State(
            self._init_state.graph,
            self._init_state.sched,
            self._init_state.sim_res,
            self._init_state.run_res,
        )
        self._Q = [self._best_state]
        self._visited = {self._best_state.graph.digest}
        self._n_generated = 0
        self._records = []
        self._early_stop_cnt = 0

    def _push(self, G: OpGraph, history=()):
        if G is None:
            return
        self._n_generated += 1
        if G.digest in self._visited:
            return
        self._visited.add(G.digest)
        state = self._State(G, *self._scheduler.run(G), history=history)
        self._Q.append(state)
        LOG.debug("push state: {}", state)

    def _run(self, G: OpGraph) -> Tuple[OpGraph, HierSched]:
        cur_iter = 0
        init_time = time.perf_counter()

        while len(self._Q) > 0:
            with self._best_state.unreal_context():
                self._Q = [
                    state
                    for state in self._Q
                    if self._f_state_key(state, d2=self._delta)
                    <= self._f_state_key(
                        self._best_state, d1=self._delta, d2=self._delta
                    )
                ]
            LOG.info("length of current search queue: {}", len(self._Q))
            Q = sorted(zip(self._Q, range(len(self._Q))))
            has_new_best = False
            for i in range(0, len(Q), utils.ceil(len(Q) / self._beam)):
                cur_state, index = Q[i]
                self._Q.pop(index)
                # cur_state, index = min(zip(self._Q, range(len(self._Q))))
                # self._Q.pop(index)
                try:
                    cur_state.run_res = self._profile(state=cur_state)
                except Exception as e:
                    LOG.exception(e)
                    continue
                LOG.info(f"pop state: {cur_state}")
                if cur_state < self._best_state:
                    self._best_state = cur_state
                    has_new_best = True
                    LOG.info(f"new best state: {cur_state}")

                for mutation, info in self._mutator.gen_mutations(
                    cur_state.graph, sim_res=cur_state.sim_res, sched=cur_state.sched
                ):
                    try:
                        self._push(mutation(), cur_state.history + (info,))
                    except Exception as e:
                        LOG.exception(e)
                        continue

            if not has_new_best:
                self._early_stop_cnt += 1
            else:
                self._early_stop_cnt = 0
            cur_iter += 1

            cur_time = time.perf_counter() - init_time
            self._records.append(
                (
                    cur_iter,
                    cur_time,
                    self._n_generated,
                    len(self._visited),
                    *self._best_state.run_res,
                    self._best_state.sim_res.latency,
                    self._best_state.sim_res.peak_memory,
                )
            )
            LOG.info(f"visited/generated: {len(self._visited)}/{self._n_generated}")
            LOG.info(f"current iter consumption: {cur_iter}/{self._iter_budget}")
            if cur_iter >= self._iter_budget:
                break
            LOG.info(f"current time consumption: {cur_time}/{self._time_budget}")
            if cur_time >= self._time_budget:
                break
            if (
                self._early_stop is not None
                and self._early_stop_cnt >= self._early_stop
            ):
                LOG.info(
                    f"{self._early_stop} iters have not update best state, early stop"
                )
                break

        LOG.info(f"final best state: {self._best_state}")
        return self._best_state.graph, self._best_state.sched
