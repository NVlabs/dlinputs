# Copyright (c) 2017 NVIDIA CORPORATION. All rights reserved.
# See the LICENSE file for licensing terms (BSD-style).

import codecs
import functools as ft
import glob
import itertools
import math
import os
import os.path
import random as pyr
import re
import sqlite3
import StringIO
import tarfile
import urllib2
import urlparse

import numpy as np
import PIL
import pylab
import scipy.ndimage as ndi
import simplejson
import zmq

from decorators import itfilter, itmapper, itsink, itsource, prints


def check_simple(v):
    if isinstance(v, (int, float, str, unicode)):
        return
    if isinstance(v, list):
        for x in v:
            check_simple(x)
    if isinstance(v, dict):
        for x, y in v.items():
            check_simple(x)
            check_simple(y)
    raise ValueError("{} {} is not a simple type".format(v, type(v)))


def dict2parts(sample, check=True):
    """Takes a sample and turns it into a list of parts.

    A sample is a dictionary of values. The dictionary is encoded
    in JSON, with any tensor values encoded as byte buffers in subsequent
    parts. This allows for efficient sending as multipart ZMQ messages.
    Although the format in principle allows for nested structures containing
    tensors, only tensors at the toplevel are encoded as buffers.
    """
    result = {}
    parts = [None]
    for k, v in sample.items():
        if isinstance(v, np.ndarray):
            a = v
            v = dict(__shape__=list(a.shape),
                     __type__=a.dtype.name,
                     __data__=len(parts))
            result[k] = v
            parts.append(np.getbuffer(a))
            continue
        if check:
            check_simple(v)
        result[k] = v
    parts[0] = simplejson.dumps(result)
    return parts


def parts2dict(parts):
    """Takes a list of parts (usually from ZMQ) and reassembles a sample.

    The `parts[0]` values must be a valid JSON string representing a dictionary.
    The `parts[1:]` values must be buffers or strings.
    Within that dictionary, values referring to tensors are replaced by
    numpy arrays pointing to the buffers in the remaining parts.
    """
    assert isinstance(parts[0], (str, unicode)), "parts[0] must be a string"
    result = simplejson.loads(parts[0])
    assert isinstance(result, dict), "json string must encode a dictionary"
    for k, v in result.items():
        if not isinstance(v, dict):
            continue
        keys = v.keys()
        if "__data__" not in keys:
            continue
        if "__shape__" not in keys:
            continue
        if "__type__" not in keys:
            continue
        assert isinstance(v["__data__"], int)
        data = parts[v["__data__"]]
        assert isinstance(data, (str, buffer))
        shape = v["__shape__"]
        assert isinstance(shape, list)
        dtype = v["__type__"]
        assert isinstance(dtype, str)
        a = np.frombuffer(data, dtype=dtype).reshape(*shape)
        result[k] = a
    return result


def zmqread(socket, nocopy=False):
    parts = socket.recv_multipart(copy=not nocopy)
    return parts2dict(parts)


def zmqwrite(socket, sample):
    parts = dict2parts(sample)
    socket.send_multipart(parts)


@itsource
def itzmq(connect=None, bind=None, kind=zmq.PULL):
    context = zmq.Context.instance()
    socket = context.socket(kind)
    if bind is not None:
        socket.bind(bind)
    elif connect is not None:
        socket.connect(connect)
    while True:
        parts = socket.recv_multipart()
        yield parts2dict(parts)


def zmqserver(data, connect=None, bind=None, kind=zmq.PUSH, rate=0.0):
    context = zmq.Context.instance()
    socket = context.socket(kind)
    if bind is not None:
        socket.bind(bind)
    elif connect is not None:
        socket.connect(connect)
    for sample in data:
        assert isinstance(sample, dict)
        parts = dict2parts(sample)
        socket.send_multipart(parts)
