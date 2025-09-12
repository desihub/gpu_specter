"""
General utility non-I/O code for gpu_specter.
"""

import time, datetime

import os, logging

import numpy as np

try:
    import cupy as cp
except ImportError:
    pass

def gather_ndarray(sendbuf, comm, root=0):
    """Gather multidimensional ndarray objects to one process from all other processes in a group.

    Args:
        sendbuf: multidimensional ndarray
        comm: mpi communicator
        root: rank of receiving process
    Returns:
        recvbuf: A stacked multidemsional ndarray if comm.rank == root, otherwise None.

    """
    rank = comm.rank
    # Save shape and flatten input array
    sendbuf = np.array(sendbuf)
    shape = sendbuf.shape
    sendbuf = sendbuf.ravel()
    # Collect local array sizes using the high-level mpi4py gather
    sendcounts = np.array(comm.gather(len(sendbuf), root))
    if rank == root:
        recvbuf = np.empty(sum(sendcounts), dtype=sendbuf.dtype)
    else:
        recvbuf = None
    comm.Gatherv(sendbuf=sendbuf, recvbuf=(recvbuf, sendcounts), root=root)
    if rank == root:
        # Reshape output before returning
        recvbuf = recvbuf.reshape((-1,) + shape[1:])
    return recvbuf

def get_array_module(x):
    """Returns the array module for arguments.

    This function is used to implement CPU/GPU generic code. If the argument
    is a :class:`cupy.ndarray` object, the :mod:`cupy` module is returned.

    For more details see: https://docs-cupy.chainer.org/en/stable/reference/generated/cupy.get_array_module.html

    Args:
        args: array to determine whether NumPy or CuPy should be used.

    Returns:
        module: :mod:`cupy` or :mod:`numpy` is returned based on the types of
        the arguments.
    """
    try:
        return cp.get_array_module(x)
    except NameError:
        # If the cupy module is unavailble, default to numpy
        return np

#- subset of desiutil.log.get_logger, to avoid desiutil dependency
_loggers = dict()
def get_logger(level=None):

    if level is None:
        level = os.getenv('DESI_LOGLEVEL', 'INFO')

    level = level.upper()
    if level == 'DEBUG':
        loglevel = logging.DEBUG
    elif level == 'INFO':
        loglevel = logging.INFO
    elif level == 'WARN' or level == 'WARNING':
        loglevel = logging.WARNING
    elif level == 'ERROR':
        loglevel = logging.ERROR
    elif level == 'FATAL' or level == 'CRITICAL':
        loglevel = logging.CRITICAL
    else:
        raise ValueError('Unknown log level {}; should be DEBUG/INFO/WARNING/ERROR/CRITICAL'.format(level))

    if level not in _loggers:
        logger = logging.getLogger('desimeter.'+level)
        logger.setLevel(loglevel)

        #- handler and formatter code adapted from
        #- https://docs.python.org/3/howto/logging.html#configuring-logging

        # create console handler and set level to debug
        ch = logging.StreamHandler()
        ch.setLevel(loglevel)

        # create formatter
        formatter = logging.Formatter('%(levelname)s:%(filename)s:%(lineno)s:%(funcName)s:%(message)s')

        # add formatter to ch
        ch.setFormatter(formatter)

        # add ch to logger
        logger.addHandler(ch)

        _loggers[level] = logger

    return _loggers[level]


class Timer(object):
    def __init__(self):
        """A helper class for capturing timing splits.

        The start time is set on instantiation.
        """
        self.start = self.time()
        self.splits = list()
        self._max_name_length = 5

    def split(self, name):
        """Capture timing split since start or previous split.

        Args:
            name: name to use for the captured interval
        """
        split = (name, self.time())
        self._max_name_length = max(len(name), self._max_name_length)
        self.splits.append(split)

    def time(self):
        """Returns the number of seconds since start of unix epoch.
        """
        return time.time()

    def _gen_split_summary(self):
        """Split summary generator.
        """
        start_iso = datetime.datetime.fromtimestamp(self.start, datetime.timezone.utc).isoformat()
        yield '{name:>{n}s}:{time}'.format(name='start', time=start_iso, n=self._max_name_length)
        last = self.start
        fmt = '{name:>{n}s}:{delta:>22.2f}'
        for name, time in self.splits:
            delta = time - last
            yield fmt.format(name=name, delta=delta, n=self._max_name_length)
            last = time
        yield fmt.format(name='total', delta=last-self.start, n=self._max_name_length)

    def log_splits(self, log):
        """Logs the timer's split summary as INFO

        Args:
            log: a logger object
        """
        for line in self._gen_split_summary():
            log.info(line)

    def print_splits(self):
        """Prints the timer's split summary
        """
        for line in self._gen_split_summary():
            print(line)
