from typing import List, Tuple, Dict, Set
from collections import defaultdict
from dataclasses import dataclass

from .op_graph import OpGraph, OpId
from .operators import FissionOp
from . import utils


@dataclass
class SimOpLife:
    start_time: float = 0
    end_time: float = float("inf")
    free_time: float = float("inf")

    def overlap(self, other: "SimOpLife"):
        if self.free_time is None or other.free_time is None:
            return False
        return not (
            self.free_time <= other.start_time or other.free_time <= self.start_time
        )

    def exec_in(self, t):
        return self.start_time <= t < self.end_time

    def live_in(self, t):
        return self.start_time <= t < self.free_time


@dataclass
class SimMemUsage:
    time_points: List[float]  # discrete-time-id to real-time
    foot_prints: List[int]  # the i-th time-point's memory footprint
    peak_op_ids: Set[OpId] = None
    time_to_idx: Dict[float, int] = None  # real-time to discrete-time-id
    peak_timeid: int = None
    peak_memory: int = None
    peak_timept: int = None
    live_op_ids: List[Set[OpId]] = None

    def __post_init__(self):
        if self.peak_timeid is None or self.peak_memory is None:
            self.peak_timeid, self.peak_memory = max(
                enumerate(self.foot_prints), key=lambda t: t[1]
            )
        if self.peak_timept is None:
            self.peak_timept = self.time_points[self.peak_timeid]
        if self.time_to_idx is None:
            self.time_to_idx = dict((t, i) for i, t in enumerate(self.time_points))
        if self.peak_op_ids is None:
            assert self.live_op_ids is not None
            self.peak_op_ids = self.live_op_ids[self.peak_timeid]


@dataclass
class SimResult:
    graph: OpGraph
    op_lifes: Dict[OpId, SimOpLife]
    latency: float = None
    _mem_usg: SimMemUsage = None

    def __post_init__(self):
        if self.latency is None:
            self.latency = max(
                self.op_lifes.values(), key=lambda life: life.end_time
            ).end_time

    @property
    def mem_usage(self):
        if self._mem_usg is None:
            self._mem_usg = self._get_mem_usage()
        return self._mem_usg

    def _get_mem_usage(self):
        pts = set()
        for life in self.op_lifes.values():
            pts.update((life.start_time, life.end_time, life.free_time))
        pts = sorted(pts)
        pt2id = dict((t, i) for i, t in enumerate(pts))
        mem = [0] * len(pts)
        for op_id, life in self.op_lifes.items():
            op = self.graph.fs_get_op(op_id)
            # tensor's lifetime is a interval：[start_time, free_time)
            if not op.is_store():
                mem[pt2id[life.start_time]] += op.out_memory + op.extra_memory
                mem[pt2id[life.end_time]] -= op.extra_memory
                mem[pt2id[life.free_time]] -= op.out_memory
        peak_i = 0
        peak_m = mem[0]
        for i in range(1, len(mem)):
            mem[i] += mem[i - 1]
            if peak_m < mem[i]:
                peak_m = mem[i]
                peak_i = i

        peak_t = pts[peak_i]
        peak_ids = {oid for oid, life in self.op_lifes.items() if life.live_in(peak_t)}
        return SimMemUsage(pts, mem, peak_ids, pt2id, peak_i, peak_m, peak_t)

    @property
    def peak_memory(self):
        return self.mem_usage.peak_memory

    def lifetime_overlap(self, a: OpId, b: OpId):
        return self.op_lifes[a].overlap(self.op_lifes[b])

    def select_swap_remat_candidates(self) -> List[Tuple[OpId, OpId, float]]:
        peak_t = self.mem_usage.peak_timept
        candidates = []
        for oid in self.mem_usage.peak_op_ids:
            op = self.graph.fs_get_op(oid)
            if op.is_mem_op():
                continue
            oid_life = self.op_lifes[oid]
            assert oid_life.live_in(peak_t)
            if oid_life.exec_in(peak_t):
                continue
            sid_lifes = [
                (sid, self.op_lifes[sid]) for sid in self.graph.fs_suc_ids(oid)
            ]
            if len(sid_lifes) <= 0:
                continue
            if any(life.exec_in(peak_t) for _, life in sid_lifes):
                continue
            sid, sid_life = min(
                [(sid, life) for sid, life in sid_lifes if life.start_time > peak_t],
                key=lambda t: t[1].start_time,
            )
            if self.graph.fs_get_op(sid).is_mem_op():
                continue
            score = (
                min(
                    sid_life.start_time - peak_t,
                    peak_t - oid_life.end_time,
                )
                * 1
            )
            candidates.append((oid, sid, score))
        return sorted(candidates, key=lambda t: t[-1], reverse=True)


class BaseSimulator:
    def __init__(self, save_last_sim_res=False) -> None:
        self._save_last_sim_res = save_last_sim_res

    def _reset(self, graph: OpGraph = None, init_time=0, op_lifes=None):
        self._graph = graph
        self._init_time = init_time
        self._op_lifes: Dict[OpId, SimOpLife] = op_lifes

    def _post_reset(self):
        pass

    def _run_one_step(self, step_id: int, op_id: OpId):
        raise NotImplementedError()

    def run(self, graph: OpGraph, init_time=0):
        self._reset(graph, init_time, defaultdict(SimOpLife))
        self._post_reset()

        for step, op_id in enumerate(self._graph.sched):
            self._run_one_step(step, op_id)
        end_time = max(self._op_lifes.values(), key=lambda life: life.end_time).end_time

        for op_id, life in self._op_lifes.items():
            if isinstance(op_id, (list, tuple)):
                continue
            op = self._graph[op_id]
            if not op.in_mem:
                assert op.is_store()
                life.free_time = life.end_time
                continue
            suc_ids = self._graph.suc_ids(op_id)
            if len(suc_ids) <= 0:
                life.free_time = end_time
            else:
                life.free_time = max(self._op_lifes[sid].end_time for sid in suc_ids)

        sim_res = SimResult(self._graph, self._op_lifes, end_time - init_time)
        self._reset()
        if self._save_last_sim_res:
            self.last_sim_res = sim_res
        return sim_res

    def _handle_fs_op(self, op_id: OpId, op: FissionOp, cur_time):
        sim_res = self.__class__().run(op.fs_graph, cur_time)
        op._latency = sim_res.latency * (op.fs_length // op.fs_factor)
        self._op_lifes.update(
            {
                (op_id, *utils.to_tuple(ids)): life
                for ids, life in sim_res.op_lifes.items()
            }
        )


class SyncSimulator(BaseSimulator):
    def _post_reset(self):
        self._cur_time = self._init_time

    def _run_one_step(self, step_id: int, op_id: OpId):
        op = self._graph[op_id]
        if op.is_fission():
            self._handle_fs_op(op_id, op, self._cur_time)
        life = self._op_lifes[op_id]
        life.start_time = self._cur_time
        life.end_time = self._cur_time + op.latency
        self._cur_time = life.end_time


class AsyncSimulator(BaseSimulator):
    def _post_reset(self):
        self._cur_times = [self._init_time, self._init_time]

    def _run_one_step(self, step_id: int, op_id: OpId):
        op = self._graph[op_id]

        stream_id = int(op.is_mem_op())
        if stream_id == 1:
            self._cur_times[0] = self._cur_times[1] = max(self._cur_times)
        cur_time = self._cur_times[stream_id]
        cur_time = max(
            [
                cur_time,
                *(
                    self._op_lifes[pre_id].end_time
                    for pre_id in self._graph.pre_ids(op_id)
                ),
            ]
        )
        if op.is_fission():
            self._handle_fs_op(op_id, op, cur_time)
        life = self._op_lifes[op_id]
        life.start_time = cur_time
        life.end_time = cur_time + op.latency
        self._cur_times[stream_id] = life.end_time


DefaultSimulator = AsyncSimulator
