# Copyright (c) 2017 NVIDIA CORPORATION. All rights reserved.
# See the LICENSE file for licensing terms (BSD-style).

import argparse
import os
import os.path
import pprint
import re
import StringIO
import sys
import tarfile
import time
import PIL
import numpy as np
import scipy
import torch
import codecs
from types import NoneType

import simplejson

def pildumps(image, format="PNG"):
    """Compress an image and return it as a string.

    Can handle float or uint8 images, rank 2 or rank 3 images.
    Float images must be in the range of 0...1

    :param numpy.ndarray image: image
    :param str format: format string accepted by PIL

    """
    result = StringIO.StringIO()
    if image.dtype in [np.dtype('f'), np.dtype('d')]:
        assert np.amin(image) > -0.001 and np.amax(image) < 1.001
        image = np.clip(image, 0.0, 1.0)
        image = np.array(image * 255.0, 'uint8')
    PIL.Image.fromarray(image).save(result, format=format)
    return result.getvalue()


def autoconvert(obj):
    """Convert objects to strings.

    This handles the most common machine learning object types:
    - numbers and strings turn into strings the obvious way
    - Torch and Numpy arrays of rank 2,3 and depth 3,4 turn into PNG arrays

    :param obj: an int, float, string, or tensor
    :returns: a string or buffer

    """
    if obj is None:
        return ""
    elif isinstance(obj, (int, float)):
        return str(obj)
    elif (isinstance(obj, scipy.ndarray) and
          (obj.ndim==2 or
           (obj.ndim==3 and obj.shape[2] in [3, 4]))):
        return pildumps(obj)
    elif (isinstance(obj, torch.FloatTensor) and
          (obj.ndimension()==2 or
           (obj.ndimension()==3 and int(obj.size(0)) in [3, 4]))):
        temp = obj.numpy()
        if temp.ndim==3: temp = temp.transpose(1, 2, 0)
        return pildumps(temp)
    elif isinstance(obj, (str, buffer)):
        return obj
    elif isinstance(obj, unicode):
        return codecs.encode(obj, "utf8")
    else:
        raise Exception("unknown object type: {}".format(type(obj)))

class TarWriter(object):
    def __init__(self, fname, converters={}, names=True, keep_meta=False, default_converter=autoconvert):
        """A class for writing dictionaries to tar files.

        :param str fname: file name for tar file (.tgz)
        :param dict converters: converters for objects to buffers
        :param dict names: dictionary mapping extensions to keys (True=use keys as extensions)
        :param bool keep_meta: keep fields starting with "_"
        :param function default_converter: converter to use if nothing else is specified (autoconvert)
        """
        self.fname = fname
        self.converters = converters
        self.names = names
        self.keep_meta = keep_meta
        self.default_converter = default_converter
        if isinstance(fname, (str, unicode)):
            self.stream = open(fname, "wb")
            self.tarstream = tarfile.open(mode="w|gz", fileobj=self.stream)
        else:
            self.stream = None
            self.tarstream = tarfile.open(fileobj=fname, mode="w:gz")

    def __enter__(self):
        """Context manager."""
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager."""
        self.finish()

    def close(self):
        """Close the tar file."""
        self.finish()

    def finish(self):
        """Close the tar file."""
        self.tarstream.close()
        if self.stream:
            self.stream.close()
            self.stream = None

    def write(self, key, obj):
        """Write a dictionary to the tar file.

        This applies the conversions and renames that the TarWriter
        was configured with.

        :param str key: basename for the tar file entry
        :param str obj: dictionary of objects to be stored
        :returns: size of the entry

        """
        total = 0
        for k in sorted(obj.keys()):
            if not self.keep_meta and k[0]=="_":
                continue
            if self.names is not True:
                if k not in self.names:
                    continue
                ext = self.names.get(k, k)
            else:
                ext = k
            converter = self.converters.get(k, self.default_converter)
            v = converter(obj[k])
            assert isinstance(v, (str, buffer)),  \
                "converter didn't yield a string: %s" % ((k, type(v)),)
            now = time.time()
            ti = tarfile.TarInfo(key + "." + ext)
            ti.size = len(v)
            ti.mtime = now
            ti.mode = 0o666
            ti.uname = "bigdata"
            ti.gname = "bigdata"
            stream = StringIO.StringIO(v)
            self.tarstream.addfile(ti, stream)
            total += ti.size
        return total


class ShardWriter(object):
    def __init__(self, prefix, converters={}, names={}, shardsize=1e9):
        """Write sharded tar files.

        :param prefix: prefix for the shards to be written (`prefix-000000.tgz` etc.)
        :param converters: dictionary of converters from values to file contents
        :param names: name rewrites of the form extension=key
        :param shardsize: size of each shard
        """
        assert self.shardsize >= 10000
        self.shardsize = shardsize

        self.converters = converters
        self.names = names

        self.tarwriter = None
        self.mcstream = None
        self.prefix = prefix
        self.total = 0
        self.shard = 0
        self.progress = True

    def __enter__(self):
        """Context manager."""
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager."""
        self.finish()

    def finish(self):
        """Close the current shard and finish up."""
        self.close_streams()

    def close(self):
        """Close the current shard and finish up."""
        self.close_streams()

    def close_streams(self):
        """Close the current shard and finish up."""
        if self.tarwriter is not None:
            self.tarwriter.finish()
            self.tarwriter = None
        if self.mcstream is not None:
            del self.mcstream

    def open_next_stream(self):
        """Finish the current stream and start a new one."""
        self.close_streams()
        name = "%s-%06d.tgz" % (self.prefix, self.shard)
        self.tarwriter = TarWriter(name)
        self.shard += 1

    def write(self, key, obj):
        """Write an object to the current tar shard.

        :param key: prefix for tar entry
        :param obj: object to be written
        :returns: number of bytes written
        """
        assert isinstance(key, str)
        assert isinstance(obj, dict)
        if self.tarwriter is None:
            self.open_next_stream()
        size = tarwriter.write(key, obj)
        assert size > 0
        self.total += size
        if self.total >= self.shardsize:
            self.open_next_stream()
            self.total = 0
        return size
