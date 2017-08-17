# Copyright (c) 2017 NVIDIA CORPORATION. All rights reserved.
# See the LICENSE file for licensing terms (BSD-style).

try:
    import torch.multiprocessing as mp
except:
    import multiprocessing as mp


def parallel_job(factory, queue, index):
    data = factory()
    for sample in data:
        queue.put(sample)


def parallelize_input(factory, nthreads=4, maxsize=256):
    queue = mp.Queue(maxsize)
    jobs = [mp.Process(target=parallel_job, args=(factory, queue, i))
            for i in range(nthreads)]
    for job in jobs:
        job.start()
    while True:
        sample = queue.get()
        sample["__queue_size__"] = queue.qsize()
        yield sample
