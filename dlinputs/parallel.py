from __future__ import absolute_import

import multiprocessing as mp
from builtins import range

from . import loadable

# Copyright (c) 2017 NVIDIA CORPORATION. All rights reserved.
# See the LICENSE file for licensing terms (BSD-style).


def _parallel_job(factory, args, queue, index):
    """Helper function to start up an input queue in a subprocess."""
    data = factory(*args)
    for sample in data:
        sample["__thread__"] = index
        queue.put(sample)


def parallelize_input(factory, args=(), nthreads=4, maxsize=256):
    """Parallelize an input queue.

    :param factory: factory function for input queue
    :param args: arguments to factory function
    :param nthreads: number of subprocesses
    :param maxsize: maximum queue size for input queue
    """
    queue = mp.Queue(maxsize)
    jobs = [mp.Process(target=_parallel_job, args=(factory, args, queue, i))
            for i in range(nthreads)]
    for job in jobs:
        job.start()
    while True:
        sample = queue.get()
        sample["__queue_size__"] = queue.qsize()
        yield sample


def _factory(fname, method):
    """Helper function for parallel_load."""
    inputs = loadable.load_input(fname)
    f = getattr(inputs, method)
    return f()


def parallel_load(fname, method="training_data", nthreads=4, maxsize=256):
    """Load the input pipeline and execute it in parallel.

    :param fname: filename containing the input pipepline (.py)
    :param method: method on ``Inputs`` object (default: training_data)
    :param nthreads: number of subprocesses
    :param maxsize: maximum queue size for input queue
    """
    return parallelize_input(_factory, (fname, method), nthreads=nthreads, maxsize=maxsize)
