import time
from absl import logging


def timer_wrapper(method):
    """Times the entry and exit from a function an logs it using absl.logging.info in milliseconds."""
    def timed(*args, **kwargs):
        time_entry = time.time()
        result = method(*args, **kwargs)
        time_exit = time.time()
        if 'log_time' in kwargs:
            process_name = kwargs.get('log_name', method.__name__.upper())
            kwargs['log_time'][process_name] = int((time_exit - time_entry) * 1000)
            logging.info('%r %2.2f ms' % (method.__name__, (time_exit - time_entry) * 1000))
        else:
            logging.info('%r %2.2f ms' % (method.__name__, (time_exit - time_entry) * 1000))
        return result

    return timed
