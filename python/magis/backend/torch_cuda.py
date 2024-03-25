import time
from collections import defaultdict
from typing import Dict
from string import digits

import torch

from ..operators import Operator, FissionOp
from ..op_graph import OpGraph
from .base import CodeGenBackend
from .. import utils


class TorchCudaBackend(CodeGenBackend):
    OOM_ERROR = torch.cuda.OutOfMemoryError

    def __init__(
        self,
        cuda_id=0,
        asynchronized=True,
        cache_file=True,
        dtype=torch.float32,
        use_torch_inductor=False,
    ) -> None:
        super().__init__(cache_file)

        if not isinstance(dtype, torch.dtype):
            dtype = str(dtype)
            base = dict(f="float", i="int", u="uint")[dtype[0]]
            bits = "".join(c for c in dtype if c in digits)
            dtype = getattr(torch, f"{base}{bits}")
        self.dtype = dtype
        self._dtype_nbytes = int("".join(c for c in str(dtype) if c in digits)) / 8

        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True

        self._asynchronized = asynchronized
        self._gpu_device = torch.device(f"cuda:{cuda_id}")
        self._cpu_device = torch.device("cpu")
        self._cpt_stream = torch.cuda.current_stream(self._gpu_device)
        self._mov_stream: torch.cuda.Stream = torch.cuda.Stream(self._gpu_device)

        self._torch_env = {
            "torch": torch,
            "F": torch.nn.functional,
            "nn": torch.nn,
            "aten": torch.ops.aten,
            "grad": torch.nn.grad,
            "store": self._async_store if self._asynchronized else self._sync_store,
            "load": self._async_load if self._asynchronized else self._sync_load,
            "wait_load": (
                self._async_wait_load if self._asynchronized else self._sync_wait_load
            ),
        }
        self._timer_env = {
            "start": torch.cuda.Event(enable_timing=True),
            "stop": torch.cuda.Event(enable_timing=True),
            "time": time,
        }

        self._input_env: Dict[str, torch.Tensor] = dict()
        self._store_env: Dict[str, torch.Tensor] = dict()
        self._event_env: Dict[str, torch.cuda.Event] = dict()

        self._prof_btcode_cache = dict()
        self._memory_limit = self._get_memory_limit()

        self._use_torch_inductor = use_torch_inductor

    def _get_memory_limit(self):
        dev_props = torch.cuda.get_device_properties(self._gpu_device)
        return (
            dev_props.total_memory - torch.cuda.memory_allocated(self._gpu_device)
        ) / self._dtype_nbytes

    def _async_store(self, dst: torch.Tensor, src: torch.Tensor):
        self._mov_stream.wait_stream(self._cpt_stream)
        with torch.cuda.stream(self._mov_stream):
            dst.copy_(src, non_blocking=True)
            src.record_stream(self._mov_stream)

    def _async_load(self, src: torch.Tensor, event: torch.cuda.Event = None):
        self._mov_stream.wait_stream(self._cpt_stream)
        with torch.cuda.stream(self._mov_stream):
            dst = src.to(self._gpu_device, non_blocking=True)
            self._mov_stream.record_event(event)
        return dst

    def _async_wait_load(self, src: torch.Tensor, event: torch.cuda.Event):
        event.synchronize()
        return src

    def _sync_store(self, dst: torch.Tensor, src: torch.Tensor):
        dst.copy_(src, non_blocking=True)

    def _sync_load(self, src: torch.Tensor, event: torch.cuda.Event = None):
        return src.to(self._gpu_device, non_blocking=True)

    def _sync_wait_load(self, src: torch.Tensor, event: torch.cuda.Event):
        return src

    def _gen_conv2d(
        self, op: Operator, o, i, w, stride, padding, dilation, layout, **_
    ) -> str:
        assert layout == "nchw"
        if isinstance(padding, str):
            ret = f"F.conv2d({i}, {w}, stride={stride}, padding='{padding}', dilation={dilation})"
        else:
            ret = f"F.conv2d({i}, {w}, stride={stride}, padding={padding}, dilation={dilation})"
        ret = f"{o} = {ret}"
        return ret

    def _gen_conv2d_bwd_inp(
        self, op: Operator, ig, og, w, stride, padding, dilation, layout, **_
    ) -> str:
        assert layout == "nchw"
        return f"{ig} = grad.conv2d_input({op.out_shape}, {w}, {og}, stride={stride}, padding={padding}, dilation={dilation})"

    def _gen_conv2d_bwd_wgt(
        self, op: Operator, wg, i, og, stride, padding, dilation, layout, **_
    ) -> str:
        assert layout == "nchw"
        return f"{wg} = grad.conv2d_weight({i}, {op.out_shape}, {og}, stride={stride}, padding={padding}, dilation={dilation})"

    def _gen_pool2d(
        self, op: Operator, y, x, kernel_size, padding, stride, layout, **_
    ) -> str:
        assert layout == "nchw"
        fn = op.tag.split(".")[-1]
        if isinstance(padding, str):
            return f"{y} = F.{fn}_pool2d({x}, kernel_size={kernel_size}, padding='{padding}', stride={stride})"
        else:
            return f"{y} = F.{fn}_pool2d({x}, kernel_size={kernel_size}, padding={padding}, stride={stride})"

    def _gen_pool2d_bwd(
        self, op: Operator, ig, og, i, kernel_size, padding, stride, layout, **_
    ) -> str:
        assert layout == "nchw"
        fn = op.tag.split(".")[-1]
        assert fn == "avg"
        return f"{ig} = aten.{fn}_pool2d_backward({og}, {i}, {kernel_size}, {stride}, {padding}, False, True, None)"

    def _gen_matmul(
        self, op: Operator, c, a, b, trans_a, trans_b, trans_c, **_
    ) -> str:
        if not trans_a and trans_b and not trans_c and len(op.inp_shapes[1]) == 2:
            ret = f"F.linear({a}, {b})"
        else:
            T = ".transpose(-1, -2)"
            ret = f"torch.matmul({a}{T if trans_a else ''}, {b}{T if trans_b else ''}){T if trans_c else ''}"
        ret = f"{c} = {ret}"
        return ret

    def _gen_flex_matmul(
        self,
        op,
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
        a_fused_shape = [
            utils.prod(op.inp_shapes[0][p0:p1])
            for p0, p1 in zip(a_pivots, a_pivots[1:])
        ]
        b_fused_shape = [
            utils.prod(op.inp_shapes[1][p0:p1])
            for p0, p1 in zip(b_pivots, b_pivots[1:])
        ]
        a = f"{a}.reshape({a_fused_shape})"
        b = f"{b}.reshape({b_fused_shape})"
        T = ".transpose(-1, -2)"
        ret = f"torch.matmul({a}{T if trans_a else ''}, {b}{T if trans_b else ''}){T if trans_c else ''}"
        ret = f"{ret}.reshape({op.out_shape})"
        ret = f"{c} = {ret}"
        return ret

    def _gen_slice(self, op: Operator, y, x, dims, starts, ends, **_) -> str:
        ranges = sorted(list(zip(dims, starts, ends)), key=lambda r: r[0])
        prev_dim = 0
        indices_str = ""
        for d, s, e in ranges:
            indices_str += f"{':,'*(d - prev_dim)}{s}:{e},"
            prev_dim = d + 1
        indices_str += "..."
        return f"{y} = {x}[{indices_str}]"

    def _gen_concat(self, op: Operator, y, *xs, dim=None, **_) -> str:
        return f"{y} = torch.cat([{','.join(xs)}], {dim})"

    def _gen_padding(self, op: Operator, y, x, pads, **_) -> str:
        return f"{y} = F.pad({x}, {tuple(p for pp in reversed(pads) for p in pp)})"

    def _gen_ewise_uni(self, op: Operator, y, x, **_) -> str:
        fn = op.tag.split(".")[-1]
        return f"{y} = aten.{fn}({x})"

    def _gen_ewise_uni_const(self, op: Operator, y, x, const, **_) -> str:
        fn = op.tag.split(".")[-1]
        return f"{y} = aten.{fn}({x}, {const})"

    def _gen_ewise_bin(self, op: Operator, c, a, b, commutative, **_) -> str:
        fn = op.tag.split(".")[-1]
        return f"{c} = aten.{fn}({a}, {b})"

    def _gen_reduce(self, op: Operator, y, x, dim, keepdim, **_) -> str:
        fn = op.tag.split(".")[-1]
        return f"{y} = torch.{fn}({x}, dim={dim}, keepdim={keepdim})"

    def _gen_softmax(self, op: Operator, y, x, dim, **_) -> str:
        return f"{y} = F.softmax({x}, dim={dim})"

    def _gen_softmax_bwd(self, op: Operator, xg, og, o, dim, **_) -> str:
        return f"{xg} = aten._softmax_backward_data({og}, {o}, {dim}, {self.dtype})"

    def _gen_layer_norm(self, op: Operator, y, x, norm_shape, **_) -> str:
        return f"{y} = aten.native_layer_norm({x}, {norm_shape}, None, None, 1e-5)"

    def _gen_layer_norm_bwd(
        self, op: Operator, xg, og, x, mean, std, norm_shape, **_
    ) -> str:
        return f"{xg} = aten.native_layer_norm_backward({og}, {x}, {norm_shape}, {mean}, {std}, None, None, (True, False, False))[0]"

    def _gen_placeholder(self, op: Operator, x, name, **_) -> str:
        return f"{x} = p_{name}"

    def _gen_interpolate(
        self, op: Operator, y, x, size, scale, mode, layout, **_
    ) -> str:
        assert layout == "nchw"
        return f"{y} = F.interpolate({x}, size={size}, scale_factor={scale}, mode='{mode}')"

    def _gen_interpolate_bwd(
        self, op: Operator, ig, og, out_size, layout, mode, **_
    ) -> str:
        assert layout == "nchw"
        assert mode in ("nearest", "bilinear", "bicubic")
        return f"{ig} = aten.upsample_{mode}2d_backward({og}, {out_size}, {op.out_shape}, None)"

    def _gen_reshape(self, op: Operator, y, x, **_) -> str:
        return f"{y} = torch.reshape({x}, {op.out_shape})"

    def _gen_permute(self, op: Operator, y, x, dims, **_) -> str:
        return f"{y} = torch.permute({x}, {dims})"

    def _gen_load(self, op: Operator, y, x, **_) -> str:
        return f"{y} = load({x})"

    def _gen_store(self, op: Operator, y, x, **_) -> str:
        return f"store({y}, {x})"

    def _gen_fission_out(self, op: Operator, y, x, index, **_) -> str:
        return f"{y} = {x}[{index}]"

    def _gen_index_out(self, op: Operator, y, x, index, **_) -> str:
        return f"{y} = {x}[{index}]"

    def _gen_fission(self, op: FissionOp, y, *xs, **_) -> str:
        fiss_id = self._n_fiss_ops
        self._n_fiss_ops += 1

        graph = op.fs_graph
        inp_ids = op.fs_inp_ids
        out_ids = op.fs_out_ids
        inp_ids = dict(zip(inp_ids, range(len(inp_ids))))
        out_ids = dict(zip(out_ids, range(len(out_ids))))
        inp_fs_dims = op.inp_fs_dims
        out_fs_dims = op.out_fs_dims

        def _tmp_f():
            for op_id in graph.all_ids():
                op = graph[op_id]
                suffix = self._id2suffix((fiss_id, op_id))
                if op.is_load():
                    self._event_env[f"event_{suffix}"] = torch.cuda.Event()
                if not op.in_mem:
                    self._store_env[f"cpu_store_{suffix}"] = torch.empty(
                        op.out_shape, dtype=self.dtype, device=self._cpu_device
                    ).pin_memory()

        _tmp_f()

        def gen(s):
            nonlocal pycode
            pycode += "\t" * self._code_indent + s + "\n"

        # declare input vars (to trigger wait_load)
        pycode = ", ".join(xs) + "\n"

        # initialize outputs
        outv = lambda i: f"tmp_{fiss_id}_{i}"
        out_decl_str = "".join(outv(i) + ", " for i in range(len(out_fs_dims)))
        out_decl_str += " = "
        for ds in out_fs_dims:
            assert len(ds) == 1
            out_decl_str += "0, " if next(iter(ds)) < 0 else "[], "
        gen(out_decl_str)

        # generate for-loop head
        itv = f"i_{fiss_id}"
        gen(f"for {itv} in range(0, {op.fs_length}, {op.fs_factor}):")

        # generate for-loop body
        self._code_indent += 1
        refcnt = defaultdict(int)
        for op_id in graph.sched:
            out_str = self._id2str((fiss_id, op_id), graph[op_id], is_out=True)

            ii = inp_ids.get(op_id, None)
            if ii is not None:
                ds = inp_fs_dims[ii]
                if len(ds) == 0:
                    gen(f"{out_str} = {xs[ii]}")
                else:
                    s = self._gen_slice(
                        None,
                        out_str,
                        f"{xs[ii]}",
                        ds,
                        [f"{itv}"] * len(ds),
                        [f"{itv}+{op.fs_factor}"] * len(ds),
                    )
                    gen(s)
            else:
                inp_strs = [
                    self._id2str((fiss_id, i), graph[i]) for i in graph.arg_ids(op_id)
                ]
                gen(self.gen_operator(graph[op_id], out_str, inp_strs))

                oi = out_ids.get(op_id, None)
                if oi is not None:
                    ds = out_fs_dims[oi]
                    assert len(ds) == 1
                    if next(iter(ds)) < 0:
                        gen(f"{outv(oi)} += {out_str}")
                    else:
                        gen(f"{outv(oi)}.append({out_str})")

                del_ids = []
                for pre_id in graph.pre_ids(op_id):
                    rc = refcnt[pre_id]
                    assert rc > 0, (pre_id, graph[pre_id].key)
                    if rc == 1 and not graph[pre_id].is_store():
                        del_ids.append(pre_id)
                    refcnt[pre_id] = rc - 1
                if len(del_ids) > 0:
                    gen(
                        f"del {','.join(self._id2str((fiss_id, di), graph[di], is_out=True) for di in del_ids)}"
                    )

            n_sucs = len(graph.suc_ids(op_id))
            if n_sucs <= 0:
                gen(f"del {out_str}")
            else:
                refcnt[op_id] = n_sucs

        self._code_indent -= 1

        # combine outputs
        out_comb_str = f"{y} = "
        for oi, ds in enumerate(out_fs_dims):
            assert len(ds) == 1
            d = next(iter(ds))
            if d < 0:
                out_comb_str += f"{outv(oi)}, "
            else:
                out_comb_str += f"torch.cat({outv(oi)}, dim={d}), "
        gen(out_comb_str)
        gen(f"del {','.join(outv(i) for i in range(len(out_fs_dims)))}")

        return pycode.strip()

    def _gen_return(self, op: Operator, y, *xs, **_) -> str:
        return f"{y} = {','.join(xs)}"

    def _compile_timer_code(self, body: str) -> str:
        pycode = "for _ in range(repeat + warmup):\n"
        pycode += "\tstart.record()\n"
        pycode += "\tfor _ in range(number):\n"
        for line in body.splitlines():
            pycode += f"\t\t{line}\n"
        pycode += "\tstop.record()\n"
        pycode += "\ttorch.cuda.synchronize()\n"
        pycode += "\tprof_times.append(start.elapsed_time(stop) / number)\n"
        btcode = compile(pycode, "timer", "exec")
        return btcode

    def _exec_timer_code(
        self, btcode, number=4, repeat=5, extra_env: dict = None, warmup=0
    ) -> float:
        if extra_env is None:
            extra_env = dict()
        exec_env = {
            **self._torch_env,
            **self._timer_env,
            "number": number,
            "repeat": repeat,
            "warmup": warmup,
            "prof_times": [],
            **extra_env,
        }
        exec(btcode, exec_env)

        prof_times = exec_env["prof_times"][warmup:]
        if len(prof_times) > 2:
            prof_times = sorted(prof_times)[1:-1]
        return sum(prof_times) / len(prof_times)

    def _get_op_key(self, op: Operator):
        if self.dtype == torch.float32:
            return op.key
        return (op.key, str(self.dtype))

    def _measure_operator_latency(self, op: Operator, number=4, repeat=5) -> float:
        k1, _ = op.key
        if not op.REUSE_PROF_CODE:
            k1 = op.key
        btcode, pycode = self._prof_btcode_cache.get(k1, (None, None))
        if btcode is None:
            pycode = self.gen_operator(
                op, "prof_y", [f"prof_x_{i}" for i in range(len(op.inp_shapes))]
            )
            btcode = self._compile_timer_code(pycode)
            self._prof_btcode_cache[k1] = btcode, pycode

        out_dev = inp_dev = self._gpu_device
        if op.is_load():
            inp_dev = self._cpu_device
        elif op.is_store():
            out_dev = self._cpu_device

        extra_env = {
            "prof_y": torch.empty(op.out_shape, dtype=self.dtype, device=out_dev),
            **{
                f"prof_x_{i}": torch.empty(s, dtype=self.dtype, device=inp_dev)
                for i, s in enumerate(op.inp_shapes)
            },
        }
        ret = self._exec_timer_code(btcode, number, repeat, extra_env)
        del extra_env
        return ret

    def reset(self, graph: OpGraph = None, bwd_id=None):
        self._graph = graph
        self._bwd_id = bwd_id

        del self._input_env, self._store_env, self._event_env
        torch.cuda.empty_cache()

        self._input_env: Dict[str, torch.Tensor] = dict()
        self._store_env: Dict[str, torch.Tensor] = dict()
        self._event_env: Dict[str, torch.cuda.Event] = dict()

        self._timer_code = None
        self._graph_func = None
        self._graph_args = None

        self._code_indent = 1
        self._done_load = set()

        self._graph_pycode = None

        self._n_fiss_ops = 0

        return self

    @staticmethod
    def _id2suffix(op_id):
        return "_".join(str(x) for x in utils.to_tuple(op_id))

    def _id2str(self, op_id, op: Operator, is_out=False):
        suffix = "_".join(str(x) for x in utils.to_tuple(op_id))
        if op.is_load() and not is_out and op_id not in self._done_load:
            self._done_load.add(op_id)
            return f"wait_load(x_{suffix}, event_{suffix})"
        return f"x_{suffix}" if op.in_mem else f"cpu_store_{suffix}"

    def _compile_graph(self):
        if self._graph_func is not None:
            return self

        for op_id in self._graph.all_ids():
            op = self._graph[op_id]
            suffix = self._id2suffix(op_id)
            if op.is_placeholder():
                self._input_env[f"p_{op.attrs['name']}"] = torch.empty(
                    op.out_shape, dtype=self.dtype, device=self._gpu_device
                )
            if op.is_load():
                self._event_env[f"event_{suffix}"] = torch.cuda.Event()
            if not op.in_mem:
                self._store_env[f"cpu_store_{suffix}"] = torch.empty(
                    op.out_shape, dtype=self.dtype, device=self._cpu_device
                ).pin_memory()

        args, self._graph_args = zip(*sorted(self._input_env.items()))
        rets = [f"x_{i}" for i in self._graph.all_out_ids()]
        pycode = f"def graph_func({','.join(args)}):\n"
        refcnt = defaultdict(int)
        for op_id in self._graph.sched:
            out_str = self._id2str(op_id, self._graph[op_id], is_out=True)
            inp_strs = [
                self._id2str(i, self._graph[i]) for i in self._graph.arg_ids(op_id)
            ]
            pycode += (
                "\t" + self.gen_operator(self._graph[op_id], out_str, inp_strs) + "\n"
            )
            del_ids = []
            for pre_id in self._graph.pre_ids(op_id):
                rc = refcnt[pre_id]
                assert rc > 0
                if rc == 1 and not self._graph[pre_id].is_store():
                    del_ids.append(pre_id)
                refcnt[pre_id] = rc - 1
            if len(del_ids) > 0:
                pycode += (
                    "\t"
                    + f"del {','.join(self._id2str(di, self._graph[di], is_out=True) for di in del_ids)}\n"
                )
            refcnt[op_id] = len(self._graph.suc_ids(op_id))

        if self._bwd_id is not None:
            out_str = self._id2str(self._bwd_id, self._graph[self._bwd_id], is_out=True)
            pycode += "\t" + f"{out_str}.sum().backward()\n"
        pycode += "\t" + f"return {','.join(rets)}" + "\n"
        pycode += "cell.append(graph_func)\n"
        cell = []
        exec(
            pycode,
            {**self._torch_env, **self._store_env, **self._event_env, "cell": cell},
        )
        self._graph_func = cell[0]
        self._graph_func.__module__ = self.__module__
        if self._use_torch_inductor:
            self._ti_graph_func = torch.compile(self._graph_func)
        self._graph_pycode = pycode
        self._timer_code = self._compile_timer_code(f"graph_func({','.join(args)})")

        return self

    def measure_graph_latency(
        self, number=4, repeat=5, warmup=0, use_torch_inductor=False
    ) -> float:
        assert not use_torch_inductor or self._use_torch_inductor
        self._compile_graph()
        extra_env = {
            **self._input_env,
            **self._store_env,
            **self._event_env,
            "graph_func": (
                self._ti_graph_func if use_torch_inductor else self._graph_func
            ),
        }
        return self._exec_timer_code(
            self._timer_code, number, repeat, extra_env, warmup
        )

    def measure_graph_peak_memory(self, use_torch_inductor=False) -> int:
        assert not use_torch_inductor or self._use_torch_inductor
        self._compile_graph()
        torch.cuda.reset_peak_memory_stats(self._gpu_device)
        (self._ti_graph_func if use_torch_inductor else self._graph_func)(
            *self._graph_args
        )
        return torch.cuda.max_memory_allocated(self._gpu_device) / self._dtype_nbytes
