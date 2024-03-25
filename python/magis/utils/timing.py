import time
from contextlib import contextmanager, suppress, nullcontext
from collections import defaultdict
from functools import wraps

from .logging import LOG

_ENABLE_TIMER = False
TIMER_RECORDS = defaultdict(list)


@contextmanager
def enable_timer(enable=True):
    global _ENABLE_TIMER
    old = _ENABLE_TIMER
    _ENABLE_TIMER = enable
    yield None
    _ENABLE_TIMER = old


def timer_wrapper(f=None, enable=True, log=True, name=None):
    def _f(f):
        if not enable:
            return f

        fun_name = name or f.__qualname__
        rec_item = TIMER_RECORDS[fun_name]

        @wraps(f)
        def wrap(*args, **kwargs):
            if not _ENABLE_TIMER:
                return f(*args, **kwargs)
            tic = time.perf_counter()
            res = f(*args, **kwargs)
            toc = time.perf_counter()
            t = toc - tic
            if log:
                LOG.debug(f"function {fun_name} took {t:.4f} seconds")
            rec_item.append(t)
            return res

        return wrap

    if f is None:
        return _f
    return _f(f)


@contextmanager
def timer_context(ctx_name="unknown-context", enable=True, log=True):
    if not (enable and _ENABLE_TIMER):
        yield None
        return
    tic = time.perf_counter()
    yield None
    toc = time.perf_counter()
    t = toc - tic
    if log:
        LOG.debug(f"context {ctx_name} took {t:.4f} seconds")
    TIMER_RECORDS[ctx_name].append(t)
