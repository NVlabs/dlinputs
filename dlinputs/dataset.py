#!/usr/bin/python

from __future__ import division
from __future__ import print_function
from __future__ import absolute_import
from past.utils import old_div
import argparse

from . import zcom
from . import gopen
from . import paths

_big = 1<<60

class BufferedSet(object):
    def __init__(self, source, fields=None, bufsize=50000, repeat=1):
        if isinstance(source, str):
            source = gopen.sharditerator(source)
        self.source = source
        self.repeat = repeat if repeat>0 else _big
        self.fields = fields
        self.buffer = [None] * bufsize
        self.counts = [_big+1] * bufsize
    def fetch(self):
        if self.fields is None:
            return self.source.next()
        else:
            sample = self.source.next()
            return tuple([sample[k] for k in self.fields])
    def __getitem__(self, index):
        if self.buffer[index] is None or self.counts[index] >= self.repeat:
            sample = self.fetch()
            self.buffer[index] = sample
            self.counts[index] = 0
        sample = self.buffer[index]
        if self.repeat>0:
            self.counts[index] += 1
        return sample
    def __len__(self):
        assert len(self.buffer)==len(self.counts)
        return len(self.buffer)
