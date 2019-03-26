#!/usr/bin/python

from __future__ import division
from __future__ import print_function
from __future__ import absolute_import
from past.utils import old_div
import argparse
import os
import sys

from . import zcom
from . import gopen
from . import paths
from . import filters

_big = 1<<60

class StreamingSet(object):
    def __init__(self,
                 source,
                 size,
                 fields=None,
                 repeat=1,
                 shuffle=0,
                 shardshuffle=True,
                 classes=None,
                 class_to_idx=None,
                 transform=None,
                 target_transform=None):
        if isinstance(source, str):
            source = gopen.sharditerator(source, shuffle=shardshuffle)
        if shuffle > 0:
            source = filters.shuffle(shuffle)(source)
        self.source = source
        self.size = size
        if isinstance(fields, str): fields = fields.split()
        self.fields = fields
        self.classes = classes
        self.class_to_idx = class_to_idx
        self.transform = transform
        self.target_transform = target_transform
    def fetch(self):
        return next(self.source)
    def __getitem__(self, index):
        sample = self.fetch()
        if os.environ.get("debug_samples","")!="":
            print(index, sample)
        if self.fields is not None:
            sample = tuple([sample[k] for k in self.fields])
        if isinstance(sample, tuple) and len(sample)==2:
            x, y = sample
            if self.transform is not None:
                x = self.transform(x)
            if self.target_transform is not None:
                y = self.target_transform(y)
            assert x is not None
            assert y is not None
            sample = (x, y)
        else:
            assert self.transform is None
            assert self.target_transform is None
        return sample
    def __len__(self):
        return self.size

class BufferedSet(object):
    def __init__(self,
                 source,
                 fields=None,
                 bufsize=50000,
                 repeat=1,
                 shuffle=0,
                 shardshuffle=True,
                 classes=None,
                 class_to_idx=None,
                 transform=None,
                 target_transform=None):
        if isinstance(source, str):
            source = gopen.sharditerator(source, shuffle=shardshuffle)
        if shuffle > 0:
            source = filters.shuffle(shuffle)(source)
        self.source = source
        self.repeat = repeat if repeat>0 else _big
        if isinstance(fields, str): fields = fields.split()
        self.fields = fields
        self.keys = [None] * bufsize
        self.buffer = [None] * bufsize
        self.counts = [_big+1] * bufsize
        self.classes = classes
        self.class_to_idx = class_to_idx
        self.transform = transform
        self.target_transform = target_transform
    def fetch(self):
        return next(self.source)
    def __getitem__(self, index):
        if self.buffer[index] is None or self.counts[index] >= self.repeat:
            sample = self.fetch()
            if os.environ.get("debug_samples","")!="":
                print(index, sample)
            self.keys[index] = sample.get("__key__", None)
            if self.fields is not None:
                sample = tuple([sample[k] for k in self.fields])
            self.buffer[index] = sample
            self.counts[index] = 0
        sample = self.buffer[index]
        if isinstance(sample, tuple) and len(sample)==2:
            x, y = sample
            if self.transform is not None:
                x = self.transform(x)
            if self.target_transform is not None:
                y = self.target_transform(y)
            assert x is not None
            assert y is not None
            sample = (x, y)
        else:
            assert self.transform is None
            assert self.target_transform is None
        if self.repeat>0:
            self.counts[index] += 1
        return sample
    def __len__(self):
        assert len(self.buffer)==len(self.counts)
        return len(self.buffer)
