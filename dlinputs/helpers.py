# Copyright (c) 2017 NVIDIA CORPORATION. All rights reserved.
# See the LICENSE file for licensing terms (BSD-style).

"""A set of helper functions for dealing uniformly with tensors and
ndarrays."""

import numpy as np

def rank(x):
    """Return the rank of the ndarray or tensor."""
    if isinstance(x, np.ndarray):
        return x.ndim
    else:
        return x.dim()

def size(x, i):
    """Return the size of dimension i."""
    if isinstance(x, np.ndarray):
        return x.shape[i]
    else:
        return x.size(i)

def shp(x):
    """Returns the shape of a tensor or ndarray as a tuple."""
    if isinstance(x, Variable):
        return tuple(x.data.size())
    elif isinstance(x, np.ndarray):
        return tuple(x.shape)
    else:
        raise ValueError("{}: unknown type".format(type(x)))

def as_nda(x):
    """Turns any tensor into an ndarray."""
    if isinstance(x, np.ndarray):
        return x
    if isinstance(x, list):
        return np.array(x)
    if isinstance(x, autograd.Variable):
        x = x.data
    raise ValueError("{}: can't convert to np.array".format(type(x)))

def reorder(batch, inp, out):
    """Reorder the dimensions of the batch from inp to out order.

    E.g. BHWD -> BDHW.
    """
    if inp is None: return batch
    if out is None: return batch
    assert isinstance(inp, str)
    assert isinstance(out, str)
    assert len(inp) == len(out), (inp, out)
    assert rank(batch) == len(inp), (rank(batch), inp)
    result = [inp.find(c) for c in out]
    # print ">>>>>>>>>>>>>>>> reorder", result
    for x in result: assert x >= 0, result
    if is_tensor(batch):
        return batch.permute(*result)
    elif isinstance(batch, np.ndarray):
        return batch.transpose(*result)
