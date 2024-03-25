from .base import *
from .taso_rules import *
from .sched_rules import *


def resolve(x):
    if isinstance(x, str):
        x = eval(x)()
    assert isinstance(x, RewriteRule)
    return x
