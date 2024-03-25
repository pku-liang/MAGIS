import math
import pickle
from contextlib import suppress, nullcontext

import xxhash

from .conv_utils import *
from .logging import *
from .timing import *
from .union_find_set import UnionFindSet

suppress
nullcontext
UnionFindSet


def to_nd_tuple(x, n=2):
    if not isinstance(x, (list, tuple)):
        x = (x,) * n
    assert len(x) == n
    return tuple(x)


def to_tuple(x):
    if not isinstance(x, (list, tuple)):
        x = (x,)
    return tuple(x)


def ceil(x):
    return int(math.ceil(x))


def floor(x):
    return int(math.floor(x))


def prod(xs):
    acc = 1
    for x in xs:
        acc *= x
    return acc


def pos_dim(x, l):
    return x if x >= 0 else l + x


def digest(x) -> bytes:
    return xxhash.xxh64_digest(x)


def factors(x):
    ret = []
    for f in range(1, int(math.sqrt(x)) + 1):
        if x % f == 0:
            ret.extend([f, x // f])
    return sorted(set(ret))


def save_pickle(path, obj):
    with open(path, "wb") as fp:
        pickle.dump(obj, fp)


def load_pickle(path):
    with open(path, "rb") as fp:
        return pickle.load(fp)


def render_dot(dot_str, name="graph", fmt="png", render=True):
    import graphviz

    dot = graphviz.Source(dot_str)
    if render:
        dot.render(cleanup=True, filename=name, format=fmt)
    return dot 
