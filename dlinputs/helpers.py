# Copyright (c) 2017 NVIDIA CORPORATION. All rights reserved.
# See the LICENSE file for licensing terms (BSD-style).

"""A set of helper functions for dealing uniformly with tensors and
ndarrays."""

import numpy as np
import torch
from torch import autograd, nn, optim
from torch.autograd import Variable
import torch.nn.functional as F
from scipy import ndimage

def novar(x):
    """Turns a variable into a tensor; does nothing for a tensor."""
    if isinstance(x, Variable):
        return x.data
    return x

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

torch_tensor_types = tuple([
    torch.Tensor,
    torch.FloatTensor, torch.IntTensor, torch.LongTensor,
    torch.cuda.FloatTensor, torch.cuda.IntTensor, torch.cuda.LongTensor
])

def is_tensor(x):
    if isinstance(x, Variable):
        x = x.data
    return isinstance(x, torch_tensor_types)

def shp(x):
    """Returns the shape of a tensor or ndarray as a tuple."""
    if isinstance(x, Variable):
        return tuple(x.data.size())
    elif isinstance(x, np.ndarray):
        return tuple(x.shape)
    elif isinstance(x, torch_tensor_types):
        return tuple(x.size())
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
    if isinstance(x, torch_tensor_types):
        return x.cpu().numpy()
    raise ValueError("{}: can't convert to np.array".format(type(x)))

def as_torch(x, single=True):
    """Converts any kind of tensor/array into a torch tensor."""
    if isinstance(x, Variable):
        return x.data
    if isinstance(x, torch_tensor_types):
        return x
    if isinstance(x, list):
        x = np.array(x)
    if isinstance(x, np.ndarray):
        if x.dtype == np.dtype("f"):
            return torch.FloatTensor(x)
        elif x.dtype == np.dtype("d"):
            if single:
                return torch.FloatTensor(x)
            else:
                return torch.DoubleTensor(x)
        elif x.dtype in [np.dtype("i"), np.dtype("int64")]:
            return torch.LongTensor(x)
        else:
            raise ValueError("{} {}: unknown dtype".format(x, x.dtype))
    raise ValueError("{} {}: unknown type".format(x, type(x)))

def typeas(x, y):
    """Make x the same type as y, for numpy, torch, torch.cuda."""
    assert not isinstance(x, Variable)
    if isinstance(y, Variable):
        y = y.data
    if isinstance(y, np.ndarray):
        return as_nda(x)
    if isinstance(x, np.ndarray):
        return as_torch(x)
    return x.type_as(y)

def sequence_is_normalized(x, eps=1e-3):
    """Check whether a batch of sequences BDL is normalized in d."""
    if isinstance(x, Variable):
        x = x.data
    assert x.dim() == 3
    marginal = x.sum(1)
    return (marginal - 1.0).abs().lt(eps).all()

def bhwd2bdhw(images, depth1=False):
    images = as_torch(images)
    if depth1:
        assert len(shp(images)) == 3, shp(images)
        images = images.unsqueeze(3)
    assert len(shp(images)) == 4, shp(images)
    return images.permute(0, 3, 1, 2)

def bdhw2bhwd(images, depth1=False):
    images = as_torch(images)
    assert len(shp(images)) == 4, shp(images)
    images = images.permute(0, 2, 3, 1)
    if depth1:
        assert images.size(3) == 1
        images = images.index_select(3, 0)
    return images

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

def assign(dest, src, inp=None, out=None):
    """Resizes the destination and copies the source."""
    src = reorder(as_torch(src), inp, out)
    if isinstance(dest, Variable):
        dest.data.resize_(*shp(src)).copy_(src)
    else:
        dest.resize_(*shp(src)).copy_(src)

