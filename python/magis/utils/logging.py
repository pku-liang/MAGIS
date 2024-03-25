import warnings

from loguru import logger

_old_showwarning = warnings.showwarning


def _new_showwarning(message, *args, **kwargs):
    logger.warning(message)
    _old_showwarning(message, *args, **kwargs)


warnings.showwarning = _new_showwarning

LOG = logger.bind(name="magis")
