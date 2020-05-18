"""
General utility non-I/O code for gpu_specter.
"""

import time, datetime

import os, logging

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
        self.start = self.time()
        self.splits = list()
        self.n = 5

    def split(self, name):
        split = (name, self.time())
        self.n = max(len(name), self.n)
        self.splits.append(split)

    def time(self):
        return time.time()

    def _gen_split_summary(self):
        start_iso = datetime.datetime.utcfromtimestamp(self.start).isoformat()
        yield '{name:>{n}s}:{time}'.format(name='start', time=start_iso, n=self.n)
        last = self.start
        fmt = '{name:>{n}s}:{delta:>22.2f}'
        for name, time in self.splits:
            delta = time - last
            yield fmt.format(name=name, delta=delta, n=self.n)
            last = time
        yield '{name:>{n}s}:{total:>22.2f}'.format(name='total', total=last-self.start, n=self.n)

    def log_splits(self, log):
        for line in self._gen_split_summary():
            log.info(line)

    def print_splits(self):
        for line in self._gen_split_summary():
            print(line)
