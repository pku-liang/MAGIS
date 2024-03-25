from typing import List, Dict, Tuple, Optional
import copy
import time

import numpy as np

from magis.op_graph import OpGraph, HierSched, OpId
from magis.simulator import SimResult
from .op_graph import OpGraph, OpId, HierSched
from .simulator import DefaultSimulator, SimResult
from . import utils


class BaseScheduler:
    def __init__(self, lat_limit: float = None, mem_limit: int = None) -> None:
        assert lat_limit is None or mem_limit is None
        self._lat_limit = lat_limit
        self._mem_limit = mem_limit
        self._sim = DefaultSimulator(save_last_sim_res=True)

    def run(self, G: OpGraph) -> Tuple[HierSched, SimResult]:
        raise NotImplementedError()


class RpoScheduler(BaseScheduler):
    def run(self, G: OpGraph) -> Tuple[HierSched, SimResult]:
        for hierG, _, _ in G.iter_fs_graphs(order="post"):
            hierG._sched = hierG.fwd_rpo_topo_order
        return G.export_hier_sched(), self._sim.run(G)


class RefineMemOpRpoScheduler(BaseScheduler):
    def __init__(
        self, lat_limit: float = None, mem_limit: int = None, adjust_load_op=True
    ) -> None:
        super().__init__(lat_limit, mem_limit)

        self._adjust_load_op = adjust_load_op

    def run(self, G: OpGraph) -> Tuple[HierSched, SimResult]:
        with utils.timer_context("schedule"):
            for hierG, _, _ in G.iter_fs_graphs(order="post"):
                hierG._sched = hierG.fwd_rpo_topo_order
            base_sched = G.export_flat_sched()

            # 抽离出所有的 store 和 load 以及 remat 节点
            main_sched = []
            store_op_ids: Dict[OpId, OpId] = dict()
            load_op_ids: Dict[OpId, Tuple[OpId, List[OpId]]] = dict()
            remat_op_ids: Dict[OpId, List[OpId]] = dict()
            for oid in base_sched:
                op = G.fs_get_op(oid)
                if op.is_store():
                    assert not op.is_remat()
                    pre_ids = G.fs_pre_ids(oid)
                    assert len(pre_ids) == 1
                    pid = pre_ids[0]
                    assert not G.fs_get_op(pid).is_mem_op()
                    for sid in G.fs_suc_ids(pid):
                        assert sid == oid or not G.fs_get_op(sid).is_mem_op()
                    store_op_ids[oid] = pid
                elif op.is_load():
                    assert not op.is_remat()
                    pre_ids = G.fs_pre_ids(oid)
                    assert len(pre_ids) == 1
                    pid = pre_ids[0]
                    assert G.fs_get_op(pid).is_store()
                    suc_ids = G.fs_suc_ids(oid)
                    load_op_ids[oid] = pid, suc_ids
                elif op.is_remat():
                    suc_ids = G.fs_suc_ids(oid)
                    remat_op_ids[oid] = suc_ids
                    assert len(suc_ids) > 0
                else:
                    main_sched.append(oid)

            # 将 remat-op 插入其最早执行的后继之前
            remat_sch_inds = sorted(
                (min(base_sched.index(sid) for sid in sids), rm_id, sids)
                for rm_id, sids in remat_op_ids.items()
            )
            for _, rm_id, sids in reversed(remat_sch_inds):
                i = min(main_sched.index(sid) for sid in sids)
                main_sched.insert(i, rm_id)

            # 将 store-op 插入其前驱之后
            store_sch_inds = sorted(
                (main_sched.index(pid), st_id) for st_id, pid in store_op_ids.items()
            )
            for i, st_id in reversed(store_sch_inds):
                main_sched.insert(i + 1, st_id)

            # 初始化 load-op 的插入位置（其最早执行的后继之前）
            load_sch_inds = sorted(
                (min(main_sched.index(sid) for sid in sids), ld_id, st_id)
                for ld_id, (st_id, sids) in load_op_ids.items()
            )
            for i, ld_id, _ in reversed(load_sch_inds):
                main_sched.insert(i, ld_id)

            G.import_flat_sched(main_sched)
        with utils.timer_context("simulate"):
            sim_res = self._sim.run(G)
            sim_res.mem_usage

        if not self._adjust_load_op or len(load_op_ids) <= 0:
            return G.export_hier_sched(), sim_res

        # 尝试去接近内存限制
        def adjust_load_ops(mem_limit):
            sim_res = self._sim.last_sim_res
            mem_usg = sim_res.mem_usage
            foot_prints = mem_usg.foot_prints.copy()
            new_sched = main_sched.copy()
            pre_si = -1
            for _, ld_id, st_id in load_sch_inds:
                beg_si = max(new_sched.index(st_id), pre_si)
                end_si = new_sched.index(ld_id)
                new_sched.pop(end_si)
                new_si = end_si

                pre_ti = mem_usg.time_to_idx[sim_res.op_lifes[ld_id].start_time]
                max_mem = foot_prints[pre_ti]
                ld_op = G.hier_get_op(ld_id)
                ld_mem = ld_op.out_memory + ld_op.extra_memory

                for i in range(end_si - 1, beg_si, -1):
                    oid = new_sched[i]
                    cur_ti = mem_usg.time_to_idx[sim_res.op_lifes[oid].end_time]
                    for ti in range(cur_ti, pre_ti):
                        max_mem = max(foot_prints[ti], max_mem)
                    if max_mem + ld_mem <= mem_limit:
                        new_si = i + 1
                        for ti in range(cur_ti, pre_ti):
                            foot_prints[ti] += ld_mem
                    else:
                        break
                    pre_ti = cur_ti
                new_sched.insert(new_si, ld_id)
                pre_si = new_si
            G.import_flat_sched(new_sched)
            return self._sim.run(G)

        if self._mem_limit is not None:
            if sim_res.peak_memory > self._mem_limit:
                adjust_load_ops(self._mem_limit)
        elif self._lat_limit is not None:
            assert self._mem_limit is None
            if sim_res.latency > self._lat_limit:
                l, r = sim_res.peak_memory, sim_res.peak_memory * 1.5
                while r > l + 1:
                    m = (l + r) / 2
                    if adjust_load_ops(m).latency <= self._lat_limit:
                        r = m
                    else:
                        l = m
                adjust_load_ops(r)

        return G.export_hier_sched(), self._sim.last_sim_res


# DEPRECATED
class IncScheduler(BaseScheduler):
    def __init__(
        self, _sched: List[int], G: OpGraph, entropy_bound: int, ignore_attr: bool
    ):
        super(IncScheduler).__init__()
        ###### Start of private parameters
        self.N = None
        self.sched: np.ndarray = None
        self.nw: np.ndarray = None
        self.ind: np.ndarray = None
        self.is_attr: np.ndarray = None
        self.num_attr: int = None
        self.non_attr_ids: np.ndarray = None

        # for partial reschedule
        self.rem_d: np.ndarray = None
        self.ref_cnt: np.ndarray = None
        self.cur_peak: int = None
        self.cur_mem: int = None
        self.entropy: int = None
        self.entropy_bound: int = None
        self.ignore_attr: bool = None
        ###### End of private parameters

        if not hasattr(G, "num_attr"):
            G.get_attr_nodes()

        self.sched = np.array(_sched, dtype=np.int32)
        self.N = G.num_ops
        assert np.max(self.sched) < self.N
        self.entropy = 0
        self.entropy_bound = entropy_bound

        self.is_attr = G.is_attr
        self.num_attr = G.num_attr
        self.non_attr_ids = G.non_attr_ids
        self.ignore_attr = ignore_attr

        tmp_nws = G.get_nws(G.non_attr_ids, ignore_attr)
        self.nw = np.zeros((self.N,), dtype=np.int32)
        for idx, _id in enumerate(G.non_attr_ids):
            self.nw[_id] = tmp_nws[idx]
        if ignore_attr:
            for _id in self.sched:
                assert not G.is_attr[_id]

        self.ind = np.zeros((self.N,), dtype=np.int32)
        for i in range(self.N):
            for fa in G.fas(i):
                if not G.is_attr[fa]:
                    self.ind[i] += 1

        self.ref_cnt = np.zeros((self.N,), dtype=np.int32)

        # self.dump()

    def dump(self):
        print("=========== Scheduler info ============")
        print("Nws:")
        for i in range(self.N):
            if not self.is_attr[i]:
                print(f"Id {i}, nw {self.nw[i]}")
        print("Attr:")
        for i in range(self.N):
            if self.is_attr[i]:
                print(i, end=" ")
        print("")
        print("=======================================")

    @property
    def _sched(self):
        return self.sched

    def assert_no_side_effect_first(self):
        self.sv_N = copy.copy(self.N)
        self.sv_sched = self.sched.copy()
        self.sv_nw = self.nw.copy()
        self.sv_ind = self.ind.copy()
        self.sv_is_attr = self.is_attr.copy()
        self.sv_num_attr = copy.copy(self.num_attr)
        self.sv_non_attr_ids = self.non_attr_ids.copy()

        # for partial reschedule
        self.sv_rem_d = self.rem_d.copy()
        self.sv_ref_cnt = self.ref_cnt.copy()
        self.sv_cur_peak = copy.copy(self.cur_peak)
        self.sv_cur_mem = copy.copy(self.cur_mem)
        self.sv_entropy = copy.copy(self.entropy)
        self.sv_entropy_bound = copy.copy(self.entropy_bound)

    def assert_no_side_effect_second(self):
        assert self.sv_N == self.N
        assert np.all(np.equal(self.sv_sched, self.sched))
        assert np.all(np.equal(self.sv_nw, self.nw))
        assert np.all(np.equal(self.sv_ind, self.ind))
        assert np.all(np.equal(self.sv_is_attr, self.is_attr))
        assert self.sv_num_attr == self.num_attr
        assert np.all(np.equal(self.sv_non_attr_ids, self.non_attr_ids))
        assert np.all(np.equal(self.sv_rem_d, self.rem_d))
        assert np.all(np.equal(self.sv_ref_cnt, self.ref_cnt))
        assert self.sv_cur_peak == self.cur_peak
        assert self.sv_cur_mem == self.cur_mem
        assert self.sv_entropy == self.entropy
        assert self.sv_entropy_bound == self.entropy_bound
        delattr(self, "sv_N")
        delattr(self, "sv_sched")
        delattr(self, "sv_nw")
        delattr(self, "sv_ind")
        delattr(self, "sv_is_attr")
        delattr(self, "sv_num_attr")
        delattr(self, "sv_non_attr_ids")
        delattr(self, "sv_rem_d")
        delattr(self, "sv_ref_cnt")
        delattr(self, "sv_cur_peak")
        delattr(self, "sv_cur_mem")
        delattr(self, "sv_entropy")
        delattr(self, "sv_entropy_bound")

    def reschedule_subgraph(
        self,
        ori_subg_ids: List[int],
        newG: OpGraph,
        new_subg_id_map: Dict[int, int],
        inp_maps: Dict[int, int],
        out_maps: Dict[int, int],
    ):
        tic = time.perf_counter()
        newG.get_attr_nodes()
        # print('num_attr =', newG.num_attr)
        # inp_maps: from origianl graph to new_subg
        # out_maps: from original graph to new_subg
        # new_subg_id_map: from new_subg to newG
        pos = np.zeros((self.N,), dtype=np.int32)
        for idx, _id in enumerate(self.sched):
            pos[_id] = idx
        L = len(self.sched) + 1
        R = int(-1)
        for _id, subg_id in inp_maps.items():
            if not self.is_attr[_id]:
                L = min(L, pos[_id])
        for _id, subg_id in out_maps.items():
            if not self.is_attr[_id]:
                R = max(R, pos[_id])
        if L > R:
            raise NotImplementedError()
        # print('L =', L, 'R =', R)
        # print('Ori schedule:', self.sched)
        # print('Ori nws:', [self.nw[_id] for _id in self.sched])
        # print('Input ids(old):', inp_maps.keys())
        # print('Output ids(old):', out_maps.keys())
        _ext = int(0)
        choice = _ext
        min_nw = self.N + 1
        while L - _ext >= 0 and _ext < 10:
            if self.nw[self.sched[L - _ext]] < min_nw and min_nw >= 10:
                min_nw = self.nw[self.sched[L - _ext]]
                choice = _ext
            elif (
                self.nw[self.sched[L - _ext]] < min_nw
                and self.nw[self.sched[L - _ext]] < 3
            ):
                min_nw = self.nw[self.sched[L - _ext]]
                choice = _ext
            _ext += 1
        L_bar = L - choice
        if min_nw != 0 and L_bar == 0:
            L_bar -= 1
        _ext = 0
        choice = _ext
        min_nw = self.N + 1
        while R + _ext < len(self.sched) and _ext < 10:
            if self.nw[self.sched[R + _ext]] < min_nw and min_nw >= 10:
                min_nw = self.nw[self.sched[R + _ext]]
                choice = _ext
            elif (
                self.nw[self.sched[R + _ext]] < min_nw
                and self.nw[self.sched[R + _ext]] < 3
            ):
                min_nw = self.nw[self.sched[R + _ext]]
                choice = _ext
            _ext += 1
        R_bar = R + choice
        if min_nw != 0 and R_bar == len(self.sched) - 1:
            R_bar += 1

        # get schedule ids
        resched_ids = list(new_subg_id_map.values())
        if L_bar == L:
            old_op_id = self.sched[L_bar]
            new_op_id = new_subg_id_map[inp_maps[old_op_id]]
            resched_ids.remove(new_op_id)
        if R_bar == R:
            old_op_id = self.sched[R_bar]
            new_op_id = new_subg_id_map[out_maps[old_op_id]]
            resched_ids.remove(new_op_id)

        ori_subg_ids = set(ori_subg_ids)
        for i in range(L_bar + 1, R_bar):
            if self.sched[i] in ori_subg_ids:
                continue
            resched_ids.append(self.sched[i])

        changed_nws = dict()
        changed_ind = dict()
        for _id, subg_id in inp_maps.items():
            new_id = new_subg_id_map[subg_id]
            if not self.is_attr[_id]:
                self.sched[pos[_id]] = new_subg_id_map[subg_id]
                changed_nws[new_id] = self.nw[_id]
                changed_ind[new_id] = self.ind[_id]
            else:
                changed_nws[new_id] = 0
                changed_ind[new_id] = 0
        for _id, subg_id in out_maps.items():
            new_id = new_subg_id_map[subg_id]
            if not self.is_attr[_id]:
                self.sched[pos[_id]] = new_id
                changed_nws[new_id] = self.nw[_id]
                changed_ind[new_id] = self.ind[_id]
            else:
                changed_nws[new_id] = 0
                changed_ind[new_id] = 0

        # for _id in new_subg_id_map.values():
        #     if newG.is_attr[_id]:
        #         self.nw[_id] = 0
        #         self.ind[_id] = 0

        actual_resched_ids = []
        for _id in resched_ids:
            if not newG.is_attr[_id]:
                actual_resched_ids.append(_id)
            # else:
            #     self.nw[_id] = 0
            #     self.ind[_id] = 0
        resched_ids = actual_resched_ids

        # print('L_bar =', L_bar)
        # print('R_bar =', R_bar)
        # print('Current schdule =', self.sched)
        # print('Resched_ids =', resched_ids)
        # print('Changed_nws =', changed_nws)
        # print('Changed_ind =', changed_ind)
        self.update(
            newG,
            resched_ids,
            changed_ind,
            changed_nws,
            None if L_bar == -1 else self.sched[L_bar],
            None if R_bar == len(self.sched) else self.sched[R_bar],
        )
        # for i in range(newG.num_ops):
        #     print(f'id: {i}, is_attr: {newG.is_attr[i]}, nw: {self.nw[i]}, ind: {self.ind[i]}')
        self.resim(newG, L_bar + 1)
        # for i in range(newG.num_ops):
        #     print(f'id: {i}, rem_ind: {self.rem_d[i]}, ref_cnt: {self.ref_cnt[i]}')
        # print(f'CurMem: {self.cur_mem}, PeakMem: {self.peak_mem}')

        # self.assert_no_side_effect_first()
        # sub_sched = self.rpo_schedule(newG, resched_ids)
        sub_sched = self._schedule(newG, resched_ids, bound=1)
        # self.assert_no_side_effect_second()

        # print('new sub_sched =', sub_sched)
        new_sched = np.zeros((newG.num_non_attr,), dtype=np.int32)
        new_sched[: L_bar + 1] = self.sched[: L_bar + 1]
        new_sched[L_bar + 1 : L_bar + 1 + sub_sched.shape[0]] = sub_sched
        new_sched[L_bar + 1 + sub_sched.shape[0] :] = self.sched[R_bar:]
        self.entropy += 1
        if self.entropy > self.entropy_bound:
            # self.nw = newG.get_nws(np.arange(self.N, dtype=np.int32))
            tmp_nws = newG.get_nws(newG.non_attr_ids, self.ignore_attr)
            self.nw = np.zeros((self.N,), dtype=np.int32)
            for idx, _id in enumerate(newG.non_attr_ids):
                self.nw[_id] = tmp_nws[idx]
            self.entropy = 0
        self.sched = new_sched
        toc = time.perf_counter()
        # print(f'New schedule = {self.sched}, Elapsed = {toc-tic}')

    def update(
        self,
        newG: OpGraph,
        resched_ids: List[int],
        changed_ind: Dict[int, int],
        changed_nws: Dict[int, int],
        L_bar_id: Optional[int],
        R_bar_id: Optional[int],
    ):
        self.N = newG.num_ops
        if self.nw.shape[0] < self.N:
            self.nw = np.concatenate(
                [self.nw, np.zeros((self.N - self.nw.shape[0],), dtype=np.int32)]
            )
        if self.ref_cnt.shape[0] < self.N:
            self.ref_cnt = np.concatenate(
                [
                    self.ref_cnt,
                    np.zeros((self.N - self.ref_cnt.shape[0],), dtype=np.int32),
                ]
            )
        if self.ind.shape[0] < self.N:
            self.ind = np.concatenate(
                [self.ind, np.zeros((self.N - self.ind.shape[0],), dtype=np.int32)]
            )
        new_nws = newG.get_nws(np.array(resched_ids, dtype=np.int32), True)

        for new_id, v in changed_ind.items():
            self.ind[new_id] = v
        for new_id, v in changed_nws.items():
            self.nw[new_id] = v

        for idx, _id in enumerate(resched_ids):
            self.nw[_id] = new_nws[idx]
        for idx, _id in enumerate(resched_ids):
            self.ind[_id] = 0
            for fa in newG.fas(_id):
                if not newG.is_attr[fa]:
                    self.ind[_id] += 1

        if L_bar_id is not None:
            self.ind[L_bar_id] = 0
            for fa in newG.fas(L_bar_id):
                if not newG.is_attr[fa]:
                    self.ind[L_bar_id] += 1
        if R_bar_id is not None:
            self.ind[R_bar_id] = 0
            for fa in newG.fas(_id):
                if not newG.is_attr[fa]:
                    self.ind[R_bar_id] += 1

        self.is_attr = newG.is_attr
        self.num_attr = newG.num_attr
        self.non_attr_ids = newG.non_attr_ids

        # for i in range(self.N):
        #     if self.is_attr[i]:
        #         for ch in newG.chs(i):
        #             if not self.is_attr[ch]:
        #                 self.ind[ch] -= 1

    def rpo_schedule(self, newG: OpGraph, resched_ids: List[int]) -> np.ndarray:
        sub_sched = []
        vis = set()
        sched_id_set = set(resched_ids)

        def dfs(now):
            vis.add(now)
            # print(now, end=' ')
            for ch in newG.chs(now):
                if not ch in vis and ch in sched_id_set:
                    dfs(ch)
            sub_sched.append(now)

        for _id in resched_ids:
            if self.rem_d[_id] == 0:
                # print(_id)
                assert _id not in vis
                dfs(_id)
                # print('')
        sub_sched.reverse()
        return np.array(sub_sched, dtype=np.int32)

    def _schedule(
        self, newG: OpGraph, resched_ids: List[int], bound: int = 10
    ) -> np.ndarray:
        subSubs = newG.split_subgraph(resched_ids, bound=bound)
        # print('Splitted subgraph =', subSubs)
        ret = np.zeros((len(resched_ids),), dtype=np.int32)
        gt0_pars = []
        gt0_peaks = []
        lt0_pars = []
        lt0_peaks = []
        for par_g in subSubs:
            sub_sched, live, peak = self.sched_seg(newG, par_g)
            if live < 0:
                lt0_pars.append(sub_sched)
                lt0_peaks.append(peak)
            else:
                gt0_pars.append(sub_sched)
                gt0_peaks.append(peak)
        gt0_peaks = np.array(gt0_peaks, dtype=np.int32)
        lt0_peaks = np.array(lt0_peaks, dtype=np.int32)
        lt0_idx = np.argsort(lt0_peaks)
        gt0_idx = np.argsort(gt0_peaks)
        pt = 0
        for i in range(lt0_idx.shape[0]):
            cur_sched = lt0_pars[lt0_idx[i]]
            ret[pt : pt + cur_sched.shape[0]] = cur_sched
            pt += cur_sched.shape[0]
        for i in range(gt0_idx.shape[0]):
            cur_sched = gt0_pars[gt0_idx[-i - 1]]
            ret[pt : pt + cur_sched.shape[0]] = cur_sched
            pt += cur_sched.shape[0]
        return ret

    # no side effect
    def sched_seg(self, newG: OpGraph, seg: List[np.ndarray]):
        L = 0
        for blk in seg:
            L += blk.shape[0]
        ret = np.zeros((L,), np.int32)
        pt = 0
        prev_live = 0
        prev_peak = int(-2147483633)

        # # check no side effect
        # sv_rem_d = self.rem_d.copy()
        # sv_ref_cnt = self.ref_cnt.copy()
        # sv_cur_mem = self.cur_mem
        # sv_peak_mem = self.peak_mem

        for blk in seg:
            ret[pt : pt + blk.shape[0]], prev_live, prev_peak = self.push_blk(
                newG, blk, prev_live, prev_peak
            )
            pt += blk.shape[0]
        for i in range(len(seg)):
            self.pop_blk(newG, seg[-i - 1])
        # assert np.max(np.abs(self.rem_d-sv_rem_d)) == 0, 'Side effects!'
        # assert np.max(np.abs(self.ref_cnt-sv_ref_cnt)) == 0, 'Side effects!'
        # assert self.cur_mem == sv_cur_mem, 'Side effects!'
        # assert self.peak_mem == sv_peak_mem, 'Side effects!'
        return ret, prev_live, prev_peak

    def dump_S(self, S: int, n: int):
        res = ""
        for i in range(n):
            if S & (1 << i):
                res = "1" + res
            else:
                res = "0" + res
        return res

    # have side effect
    def push_blk(self, newG: OpGraph, blk: np.ndarray, prev_live: int, prev_peak: int):
        blk_set = set()
        for i in range(blk.shape[0]):
            blk_set.add(blk[i])
        n = blk.shape[0]
        tot = 1 << n

        def __push(S: int):
            live = prev_live
            for i in range(blk.shape[0]):
                opid = blk[i]
                if S & (1 << i):
                    chs = newG.chs(opid)
                    self.ref_cnt[opid] = len(chs)
                    live += newG[opid].out_memory
                    for ch in chs:
                        assert not newG.is_attr[ch]
                        self.rem_d[ch] -= 1
            for i in range(blk.shape[0]):
                if S & (1 << i):
                    if self.rem_d[blk[i]] != 0:
                        print(i, blk[i])
                    assert self.rem_d[blk[i]] == 0
            for i in range(blk.shape[0]):
                if S & (1 << i):
                    for fa in newG.fas(blk[i]):
                        faop = newG[fa]
                        if not newG.is_attr[fa]:
                            self.ref_cnt[fa] -= 1
                            if self.ref_cnt[fa] == 0 and faop.in_mem:
                                live -= newG[fa].out_memory
            return live

        def __pop(S: int):
            for i in range(blk.shape[0]):
                if S & (1 << i):
                    op_id = blk[i]
                    for fa in newG.fas(op_id):
                        if not newG.is_attr[fa]:
                            self.ref_cnt[fa] += 1
            for i in range(blk.shape[0]):
                if S & (1 << i):
                    op_id = blk[i]
                    self.ref_cnt[op_id] = 0
                    for ch in newG.chs(op_id):
                        assert not newG.is_attr[ch]
                        self.rem_d[ch] += 1
            return prev_live

        def __check_valid(S: int):
            ret = True
            for i in range(n):
                if S & (1 << i):
                    opid = blk[i]
                    for fa in newG.fas(opid):
                        for j in range(blk.shape[0]):
                            if blk[j] == fa:
                                if not (S & (1 << j)):
                                    ret = False
                                break
                        if not ret:
                            break
                if not ret:
                    break
            return ret

        def verify_push_pop(tot: int):
            for S in range(tot):
                if not __check_valid(S):
                    continue
                sv_ref_cnt = self.ref_cnt.copy()
                sv_rem_d = self.rem_d.copy()
                _live = __push(S)
                # print(self.dump_S(S, blk.shape[0]), _live)
                __pop(S)
                assert np.all(np.equal(sv_ref_cnt, self.ref_cnt))
                assert np.all(np.equal(sv_rem_d, self.rem_d))

        if blk.shape[0] > 20:
            print(blk.shape[0])
            raise NotImplementedError()

        Peak = np.ones((tot,), dtype=np.int32) * int(2147483633)
        Live = np.ones((tot,), dtype=np.int32) * (-int(2147483633))
        From = np.ones((tot,), dtype=np.int32) * (-1)
        Peak[0] = prev_peak
        Live[0] = prev_live

        # verify_push_pop(tot)

        cnt_active_state = 0
        for S in range(tot):
            if Peak[S] != int(2147483633):
                cnt_active_state += 1
                assert Live[S] != -int(2147483633)
                __push(S)
                for i in range(n):
                    opid = blk[i]
                    newop = newG[opid]
                    if (not (S & (1 << i))) and self.rem_d[opid] == 0:
                        newS = S | (1 << i)
                        newPeak = max(
                            Live[S] + newop.out_memory + newop.extra_memory, Peak[S]
                        )
                        if newPeak < Peak[newS]:
                            Peak[newS] = newPeak
                            From[newS] = i
                        if Live[newS] == (-int(2147483633)):
                            Live[newS] = Live[S] + (
                                newop.out_memory if newop.in_mem else 0
                            )
                            for fa in newG.fas(opid):
                                faop = newG[fa]
                                if not newG.is_attr[fa]:
                                    if self.ref_cnt[fa] == 1 and faop.in_mem:
                                        Live[newS] -= faop.out_memory
                        else:  # verifying correctness
                            checkLive = Live[S] + (
                                newop.out_memory if newop.in_mem else 0
                            )
                            for fa in newG.fas(opid):
                                faop = newG[fa]
                                if not newG.is_attr[fa]:
                                    if self.ref_cnt[fa] == 1 and faop.in_mem:
                                        checkLive -= faop.out_memory

                            assert checkLive == Live[newS]
                __pop(S)

        ret_sched = np.zeros((n,), dtype=np.int32)
        cur_state = tot - 1
        for j in range(n, 0, -1):
            ret_sched[j - 1] = blk[From[cur_state]]
            cur_state ^= 1 << From[cur_state]
        # # first verify push pop with rpo
        # ret_sched = self.rpo_schedule(newG, blk)

        # print('    partial sched =', ret_sched)
        # print('    -- encountered', cnt_active_state, 'active state')

        cur_live = prev_live
        cur_peak = prev_peak
        for i in range(ret_sched.shape[0]):
            opid = ret_sched[i]
            assert self.rem_d[opid] == 0, "Something wrong"
            op = newG[opid]
            if op.in_mem:
                cur_live += op.out_memory + op.extra_memory
            cur_peak = max(cur_peak, cur_live)
            cur_live -= op.extra_memory
            for fa in newG.fas(opid):
                faop = newG[fa]
                if not newG.is_attr[fa]:
                    self.ref_cnt[fa] -= 1
                    if self.ref_cnt[fa] == 0 and faop.in_mem:
                        cur_live -= faop.out_memory
            chs = newG.chs(opid)
            self.ref_cnt[opid] = len(chs)
            for ch in chs:
                assert not newG.is_attr[ch]
                self.rem_d[ch] -= 1
            if not op.in_mem:
                cur_live -= op.out_memory
        # print(cur_peak, Peak[tot-1])
        assert cur_live == Live[tot - 1]
        assert cur_peak == Peak[tot - 1]

        return ret_sched, cur_live, cur_peak

    def pop_blk(self, newG: OpGraph, blk: np.ndarray):
        for i in range(blk.shape[0]):
            op_id = blk[i]
            for fa in newG.fas(op_id):
                if not newG.is_attr[fa]:
                    self.ref_cnt[fa] += 1
        for i in range(blk.shape[0]):
            op_id = blk[i]
            self.ref_cnt[op_id] = 0
            for ch in newG.chs(op_id):
                assert not newG.is_attr[ch]
                self.rem_d[ch] += 1

    def resim(self, newG: OpGraph, L: int):
        self.rem_d = self.ind.copy()
        # no need to change ref_cnt
        self.ref_cnt[:] = 0
        self.cur_mem = 0
        self.peak_mem = 0

        for i in range(L):
            op_id = self.sched[i]
            assert self.rem_d[op_id] == 0, "Something wrong"
            op = newG[op_id]

            self.cur_mem += op.out_memory + op.extra_memory
            self.peak_mem = max(self.peak_mem, self.cur_mem)

            self.cur_mem -= op.extra_memory
            for pre_id in newG.fas(op_id):
                if newG.is_attr[pre_id]:
                    continue
                pre = newG[pre_id]
                rc = self.ref_cnt[pre_id]
                assert rc > 0
                rc -= 1
                self.ref_cnt[pre_id] = rc
                if rc == 0 and pre.in_mem:
                    self.cur_mem -= pre.out_memory
            self.ref_cnt[op_id] = newG.num_chs(op_id)
            for ch in newG.chs(op_id):
                self.rem_d[ch] -= 1
            if not op.in_mem:
                self.cur_mem -= op.out_memory
