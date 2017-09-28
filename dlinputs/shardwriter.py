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

    Can handle float or uint8 images.
    """
    result = StringIO.StringIO()
    if image.dtype in [np.dtype('f'), np.dtype('d')]:
        assert np.amin(image) > -0.001 and np.amax(image) < 1.001
        image = np.clip(image, 0.0, 1.0)
        image = np.array(image * 255.0, 'uint8')
    PIL.Image.fromarray(image).save(result, format=format)
    return result.getvalue()


class ShardWriter(object):
    def __init__(self, prefix, converters={}, names={}, minio=False, shardsize=1e9):
        self.shardsize = shardsize
        self.converters = converters
        self.names = names
        self.tarstream = None
        self.mcstream = None
        self.minio = minio
        self.prefix = prefix
        self.total = 0
        self.shard = 0
        self.progress = True

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.finish()

    def close_streams(self):
        if self.tarstream is not None:
            self.tarstream.close()
            self.tarstream = None
        if self.mcstream is not None:
            self.mcstream.close()
            self.mcstream = None

    def open_next_stream(self):
        self.close_streams()
        if self.shardsize > 0:
            name = "%s-%06d.tgz" % (self.prefix, self.shard)
        else:
            name = "%s.tgz" % (self.prefix,)
        if self.minio:
            if self.progress:
                cmd = "dd status=progress | mc pipe %s" % name
            else:
                cmd = "mc pipe %s" % name
            print "# piping", cmd
            self.mcstream = os.popen(cmd, "wb")
            self.tarstream = tarfile.open(mode="w|gz", fileobj=mcstream)
        else:
            print "# writing", name
            self.tarstream = tarfile.open(name, "w:gz")
        self.shard += 1

    def write(self, key, obj):
        assert isinstance(key, str)
        assert isinstance(obj, dict)
        if self.tarstream is None:
            self.open_next_stream()
        for k in sorted(obj.keys()):
            if k not in self.names:
                continue
            ext = self.names.get(k, k)
            if k in self.converters:
                v = self.converters[k](obj[k])
            elif obj[k] is None:
                v = ""
            elif isinstance(obj[k], (int, float)):
                v = str(obj[k])
            elif isinstance(obj[k], scipy.ndarray) and ext.endswith("png"):
                assert obj[k].ndim in  [2, 3], obj[k].shape
                v = pildumps(obj[k])
            elif isinstance(obj[k], torch.FloatTensor) and ext.endswith("png"):
                temp = obj[k].numpy()
                if temp.ndim==3: temp = temp.transpose(1, 2, 0)
                assert temp.ndim in  [2, 3], obj[k].size()
                v = pildumps(obj[k])
            elif isinstance(obj[k], (str, buffer)):
                v = obj[k]
            elif isinstance(obj[k], unicode):
                v = codecs.encode(obj[k], "utf8")
            else:
                raise Exception("unknown object type: {}".format(type(obj[k])))
            assert isinstance(v, (str, unicode, buffer)), (k, type(v),)
            now = time.time()
            ti = tarfile.TarInfo(key + "." + ext)
            ti.size = len(v)
            ti.mtime = now
            ti.mode = 0o666
            ti.uname = "bigdata"
            ti.gname = "bigdata"
            stream = StringIO.StringIO(v)
            self.tarstream.addfile(ti, stream)
            self.total += ti.size
        if self.shardsize > 0 and self.total >= self.shardsize:
            self.open_next_stream()
            self.total = 0

    def finish(self):
        self.close_streams()
