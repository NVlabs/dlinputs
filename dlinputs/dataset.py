#!/usr/bin/python

from __future__ import absolute_import, division, print_function

import argparse
import os
import sys

from past.utils import old_div

from . import filters, gopen, paths, zcom

_big = 1 << 60

def getfirst(a, keys, default=None):
    for k in keys:
        result = a.get(k)
        if result is not None: return result
    return default

def transform_with(sample, transformers):
    assert len(transformers) <= len(sample)
    result = list(sample)
    for i, f in enumerate(transformers):
        result[i] = f(sample[i])
    return result

class StreamingSet(object):
    def __init__(self,
                 source,
                 size,
                 fields=None,
                 shuffle=0,
                 shardshuffle=None,
                 classes=None,
                 class_to_idx=None,
                 decoders=[],
                 transformers=[],
                 converters=[],
                 decode=True,
                 epochs=999999999):
        if shardshuffle is None:
            shardshuffle = (shuffle > 0)
        if isinstance(source, str):
            source = gopen.sharditerator(source, shuffle=shardshuffle, epochs=epochs, decode=decode)
        if shuffle > 0:
            source = filters.shuffle(shuffle)(source)
        self.source = source
        self.size = size
        if isinstance(fields, str):
            fields = fields.split()
        self.fields = [field.split(";") for field in fields]
        self.classes = classes
        self.class_to_idx = class_to_idx
        self.decoders = decoders
        self.transformers = transformers
        self.converters = converters

    def fetch(self):
        return next(self.source)

    def extract(self, sample):
        sample = [getfirst(sample, field) for field in self.fields]
        sample = transform_with(sample, self.decoders)
        sample = transform_with(sample, self.transformers)
        sample = transform_with(sample, self.converters)
        return tuple(sample)

    def __getitem__(self, index):
        sample = self.fetch()
        sample = self.extract(sample)
        return sample

    def __len__(self):
        return self.size


class BufferedSet(object):
    def __init__(self,
                 source,
                 bufsize=50000,
                 fields=None,
                 repeat=1,
                 shuffle=0,
                 shardshuffle=True,
                 classes=None,
                 class_to_idx=None,
                 decode=True,
                 decoders=[],
                 transformers=[],
                 converters=[]):
        if isinstance(source, str):
            source = gopen.sharditerator(source, shuffle=shardshuffle, decode=decode)
        if shuffle > 0:
            source = filters.shuffle(shuffle)(source)
        self.source = source
        self.repeat = repeat if repeat > 0 else _big
        if isinstance(fields, str):
            fields = fields.split()
        self.fields = [field.split(";") for field in fields]
        self.buffer = [None] * bufsize
        self.counts = [_big+1] * bufsize
        self.classes = classes
        self.decoders = decoders
        self.transformers = transformers
        self.converters = converters

    def fetch(self):
        return next(self.source)

    def extract(self, sample):
        sample = [getfirst(sample, field) for field in self.fields]
        sample = transform_with(sample, self.decoders)
        sample = transform_with(sample, self.transformers)
        sample = transform_with(sample, self.converters)
        return tuple(sample)

    def __getitem__(self, index):
        if self.buffer[index] is None or self.counts[index] >= self.repeat:
            sample = self.fetch()
            self.buffer[index] = sample
            self.counts[index] = 0
        else:
            sample = self.buffer[index]
        if self.repeat > 0:
            self.counts[index] += 1
        sample = self.extract(sample)
        return sample

    def __len__(self):
        assert len(self.buffer) == len(self.counts)
        return len(self.buffer)

class StreamingLoader(object):
    def __init__(self,
                 source,
                 size,
                 fields=None,
                 batch_size=0,
                 shuffle=0,
                 shardshuffle=None,
                 decoders=[],
                 transformers=[],
                 converters=[],
                 decode=True,
                 epochs=1,
                 verbose=False):
        self.shuffle = shuffle
        self.shardshuffle = shardshuffle if shardshuffle is not None else shuffle
        self.source = source
        self.size = size
        if isinstance(fields, str):
            fields = fields.split()
        self.fields = [field.split(";") for field in fields]
        self.epochs = epochs
        self.decode = decode
        self.decoders = decoders
        self.transformers = transformers
        self.converters = converters
        self.batch_size = batch_size
        self.verbose = verbose

    def fetch(self):
        return next(self.source)

    def extract(self, sample):
        return tuple(sample)

    def __iter__(self):
        if isinstance(self.source, str):
            source = gopen.sharditerator(self.source, shuffle=self.shardshuffle, epochs=self.epochs, decode=self.decode)
        else:
            source = iter(self.source)
        if self.shuffle > 0:
            source = filters.shuffle(self.shuffle)(source)
        if self.batch_size > 0:
            source = filters.batched(self.batch_size)(source)
        for sample in source:
            self.last_raw_sample = sample
            sample = [getfirst(sample, field) for field in self.fields]
            sample = transform_with(sample, self.transformers)
            self.last_sample = sample
            yield tuple(sample)

    def __len__(self):
        return self.size
