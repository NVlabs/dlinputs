from __future__ import absolute_import, division, print_function

import itertools
import logging
import math
import os
import pickle
import random as pyr
import re
import shelve
import tempfile
from builtins import range, str, zip
from functools import reduce, wraps

import numpy as np
from past.utils import old_div

from . import improc, sqlshelve, utils
from .gopen import *

# Copyright (c) 2017 NVIDIA CORPORATION. All rights reserved.
# See the LICENSE file for licensing terms (BSD-style).


def curried(f):
    """A decorator for currying functions in the first argument."""
    @wraps(f)
    def wrapper(*args, **kw):
        def g(x):
            return f(x, *args, **kw)
        return g
    return wrapper


def compose2(f, g):
    """Compose two functions, g(f(x))"""
    return lambda x: g(f(x))


def compose(*args):
    """Compose a sequence of functions (left-to-right)"""
    return reduce(compose2, args)


def pipeline(source, *args):
    """Write an input pipeline; first argument is source, rest are filters."""
    if len(args) == 0:
        return source
    return compose(*args)(source)


def merge(sources, weights=None):
    """Merge samples from multiple sources into a single iterator.

    :param sources: list of iterators
    :param weights: weights for sampling
    :returns: iterator

    """
    assert weights is None, "weighted sampling not implemented yet"
    while len(sources) > 0:
        index = pyr.randint(0, len(sources)-1)
        try:
            sample = next(sources[index])
            yield sample
        except StopIteration:
            del sources[index]
    raise StopIteration()


def concat(sources, maxepoch=1):
    """Concatenate multiple sources, usually for test sets.

    :param sources: list of iterators
    :param maxepochs: number of epochs (default=1)
    :returns: iterator

    """
    count = 0
    for source in sources:
        for sample in source:
            if maxepoch is not None and "__epoch__" in sample:
                if sample["__epoch__"] >= maxepoch:
                    break
            sample = dict(sample)
            sample["__count__"] = count
            yield sample
            count += 1


def objhash(obj):
    import hashlib
    obj = pickle.dumps(obj, -1)
    m = hashlib.md5()
    m.update(obj)
    return m.hexdigest()


@curried
def unique(data, key, rekey=False, skip_missing=False, error=True):
    """Ensure that data is unique in the given key.

    :param key: sample key to be made unique
    :param rekey: if True, use the hash value as the new key
    """
    finished = set()
    for sample in data:
        assert key in sample
        ident = objhash(sample.get(key))
        if ident in finished:
            if error:
                raise Exception("duplicate key")
            else:
                continue
        finished.add(ident)
        if rekey:
            sample["__key__"] = ident
        yield sample


@curried
def patched(data, patches, maxpatches=10000):
    """Patch a dataset with another dataset.

    Patches are stored in memory; for larger patch sizes, use diskpatched.

    :param patches: iterator yielding patch samples
    :param maxpatches: maximum number of patches to load
    :returns: iterator

    """
    patchdict = {}
    for i, sample in enumerate(patches):
        key = sample["__key__"]
        assert key not in patchdict, "{}: repeated key".format(key)
        assert i < maxpatches, "too many patches; increase maxpatches="
        patchdict[key] = sample
    for sample in data:
        key = sample["__key__"]
        return patchdict.get(key, sample)


@curried
def identity(data):
    for sample in data:
        yield sample


@curried
def info(data, every=0):
    """Print info about samples.

    By default only prints the first sample, but with
    `every>0`, prints `every` samples.

    :param data: sample iterator
    :param every: how often to print information
    :returns: iterator

    """
    count = 0
    for sample in data:
        if (count == 0 and every == 0) or (every > 0 and count % every == 0):
            print("# itinfo", count)
            utils.print_sample(sample)
        count += 1
        yield sample


@curried
def sliced(data, *args):
    return itertools.islice(data, *args)


@curried
def grep(source, **kw):
    """Select samples from the source that match given patterns.

    Arguments are of the form `fieldname="regex"`.
    If `_not=True`, samples matching the pattern are rejected.

    :param source: iterator
    :param kw: fieldname="regex" entries
    :returns: iterator

    """
    _not = not not kw.get("_not", False)
    if "_not" in kw:
        del kw["_not"]
    for item in source:
        for data in source:
            skip = False
            for k, v in list(kw.items()):
                matching = not not re.search(v, data[k])
                if matching == _not:
                    skip = True
                    break
            if skip:
                continue
            yield data


@curried
def select(source, **kw):
    """Select samples from the source that match given predicates.

    Arguments are of the form `fieldname=predicate`.

    :param source: iterator
    :param kw: fieldname=predicate selectors
    :returns: iterator

    """
    for data in source:
        skip = False
        for k, f in list(kw.items()):
            matching = not not f(data[k])
            if not matching:
                skip = True
                break
        if skip:
            continue
        yield data


def find_key(sample, keys):
    for k in keys.split():
        if k in sample:
            return k
    return None


def has_key(sample, keys):
    return find_key(sample, keys) is not None


@curried
def with_keys(data, *args):
    for sample in data:
        has_all_keys = all(has_key(sample, k) for k in args)
        if not has_all_keys:
            continue
        yield sample


@curried
def ren(data, kw, keep_all=False, keep_meta=True, skip_missing=False, error_missing=True):
    """Rename and select fields using new_name="old_name" keyword arguments.

    :param data: iterator
    :param keep_all: keep all fields, even those that haven't been renamed
    :param keep_meta: keep metadata (fields starting with "_")
    :param skip_missing: skip records where not all fields are present
    :param kw: new_name="old_name" rename rules
    :returns: iterator

    """
    for sample in data:
        if keep_all:
            result = dict(sample)
        else:
            result = dict(__key__=sample.get("__key__"))
        has_all_keys = all(has_key(sample, k) for k in list(kw.values()))
        if error_missing and not has_all_keys:
            raise ValueError("missing keys, got {}, want {}".format(
                list(sample.keys()), kw))
        if skip_missing and not has_all_keys:
            continue
        if keep_meta:
            for k, v in list(sample.items()):
                if k[0] == "_":
                    result[k] = v
        for k, keys in list(kw.items()):
            key = find_key(sample, keys)
            if key is None:
                continue
            result[k] = sample[key]
        yield result


def rename(keep_all=False, keep_meta=True, skip_missing=False, error_missing=True, **kw):
    return ren(kw, keep_all=keep_all, keep_meta=keep_meta, skip_missing=skip_missing, error_missing=error_missing)


def copy(keep_meta=True, skip_missing=False, error_missing=True, **kw):
    return ren(kw, keep_all=True, keep_meta=keep_meta, skip_missing=skip_missing, error_missing=error_missing)


@curried
def extract(data, *fields, **kw):
    """Extract the given fields and return a tuple.
    """
    error = False
    for sample in data:
        if kw.get("error_missing", True):
            result = [sample[f] for f in fields]
        else:
            result = [sample.get(f) for f in fields]
        yield tuple(result)

@curried
def annotate(data, table, key="__key__"):
    """Annotate samples with data from another table.
    """
    for sample in data:
        result = dict(sample)
        k = sample[key]
        extra = table.get(k, {})
        result.update(extra)
        yield result

@curried
def map(data, error_missing=True, errors_are_fatal=False, **kw):
    """Map the fields in each sample using name=function arguments.

    Unmentioned fields are left alone.

    :param data: iterator
    :param kw: name=function pairs, applying function to each sample field
    :returns: iterator

    """
    error = False
    for sample in data:
        sample = sample.copy()
        for k, f in list(kw.items()):
            if error_missing and k not in sample:
                print("map", kw, "key", k,
                      "missing from sample", list(sample.keys()))
                error = True
                break
            if error:
                break
            try:
                sample[k] = f(sample[k])
            except Exception as e:
                logging.warn("map {}".format(repr(e)))
                if errors_are_fatal:
                    print(e)
                    error = True
                    break
                sample = None
                break
        if sample is not None:
            yield sample


@curried
def check(data, **kw):
    """Apply all the check given in the keywords.
    """
    error = False
    for sample in data:
        for k, f in list(kw.items()):
            assert f(sample[k]), sample
        yield sample

@curried
def encode(data):
    """Automatically encode data items based on key extension.

    Known extensions:
    - png, jpg, jpeg: images
    - json: JSON
    - pyd, pickle: Python pickles
    - mp: Messagepack
    """
    for sample in data:
        yield utils.autoencode(data)


@curried
def decode(data):
    """Automatically decode data items based on key extension.

    Known extensions:
    - png, jpg, jpeg: images
    - json: JSON
    - pyd, pickle: Python pickles
    - mp: Messagepack
    """
    for sample in data:
        yield utils.autodecode(sample)


@curried
def transform(data, f=None):
    """Map entire samples using the given function.

    :param data: iterator
    :param f: function from samples to samples
    :returns: iterator over transformed samples

    """

    if f is None:
        def f(x): return x
    for sample in data:
        result = f(sample)
        result["__key__"] = sample.get("__key__")
        yield result

###
# Shuffling
###


@curried
def shuffle(data, bufsize=1000, initial=100):
    """Shuffle the data in the stream.

    This uses a buffer of size `bufsize`. Shuffling at
    startup is less random; this is traded off against
    yielding samples quickly.

    :param data: iterator
    :param bufsize: buffer size for shuffling
    :returns: iterator

    """
    assert initial <= bufsize
    buf = []
    startup = True
    for sample in data:
        if len(buf) < bufsize:
            buf.append(next(data))
        k = pyr.randint(0, len(buf) - 1)
        sample, buf[k] = buf[k], sample
        if startup and len(buf) < initial:
            buf.append(sample)
            continue
        startup = False
        yield sample
    for sample in buf:
        yield sample


@curried
def diskshuffle(data, bufsize=1000, initial=100, fname=None):
    """Shuffle the data in the stream.

    This uses a buffer of size `bufsize`. Shuffling at
    startup is less random; this is traded off against
    yielding samples quickly. Data is buffered on disk
    in a dbm-style database.

    :param data: iterator
    :param bufsize: buffer size for shuffling
    :returns: iterator

    """
    raise Exception("unimplemented")  # FIXME

###
# Batching
###


@curried
def batched(data, batchsize=20, combine_tensors=True, combine_scalars=True, partial=True, expand=False):
    """Create batches of the given size.

    :param data: iterator
    :param batchsize: target batch size
    :param tensors: automatically batch lists of ndarrays into ndarrays
    :param partial: return partial batches
    :returns: iterator

    """
    batch = []
    for sample in data:
        if len(batch) >= batchsize:
            yield utils.samples_to_batch(batch,
                                         combine_tensors=combine_tensors,
                                         combine_scalars=combine_scalars,
                                         expand=expand)
            batch = []
        batch.append(sample)
    if len(batch) == 0:
        return
    elif len(batch) == batchsize or partial:
        yield utils.samples_to_batch(batch,
                                     combine_tensors=combine_tensors,
                                     combine_scalars=combine_scalars,
                                     expand=expand)


def maybe_index(v, i):
    """Index if indexable.

    :param v: object to be indexed
    :param i: index
    :returns: v[i] if that succeeds, otherwise v

    """
    try:
        return v[i]
    except:
        return v


@curried
def unbatch(data):
    """Unbatch data.

    :param data: iterator over batches
    :returns: iterator over samples

    """
    for sample in data:
        keys = list(sample.keys())
        bs = len(sample[keys[0]])
        for i in range(bs):
            yield {k: maybe_index(v, i) for k, v in list(sample.items())}

###
# Image data augmentation
###


@curried
def distort(sample, distortions=[(5.0, 5)], keys=["image"]):
    """Apply distortions to images in sample.

    :param sample
    :param distortions: distortion parameters
    :param keys: fields to distort
    :returns: distorted sample

    """
    images = [sample[k] for k in keys]
    distorted = improc.random_distortions(images, distortions)
    result = dict(sample)
    for k, v in zip(keys, distorted):
        result[k] = v
    return result


@curried
def standardized(data, size, keys=["image"], crop=0, mode="nearest"):
    """Standardize images in a sample.

    :param sample: iterator over samples
    :param size: target size
    :param keys: keys for images to be distorted
    :param crop: whether to crop (1) or extend (0)
    :param mode: boundary mode
    :returns: iterator over standardized samples

    """
    if isinstance(keys, str):
        keys = keys.split(",")
    for sample in data:
        result = dict(sample)
        for key in keys:
            result[key] = improc.standardize(
                sample[key], size, crop=crop, mode=mode)
        yield result


@curried
def batchedbuckets(data, batchsize=5, scale=1.8, seqkey="image", batchdim=1):
    """List-batch input samples into similar sized batches.

    :param data: iterator of samples
    :param batchsize: target batch size
    :param scale: spacing of bucket sizes
    :param seqkey: input key to use for batching
    :param batchdim: input dimension to use for bucketing
    :returns: batches consisting of lists of similar sequence lengths

    """
    buckets = {}
    for sample in data:
        seq = sample[seqkey]
        l = seq.shape[batchdim]
        r = int(math.floor(old_div(math.log(l), math.log(scale))))
        batched = buckets.get(r, {})
        for k, v in list(sample.items()):
            if k in batched:
                batched[k].append(v)
            else:
                batched[k] = [v]
        if len(batched[seqkey]) >= batchsize:
            batched["_bucket"] = r
            yield batched
            batched = {}
        buckets[r] = batched
    for r, batched in list(buckets.items()):
        if batched == {}:
            continue
        batched["_bucket"] = r
        yield batched


###
# various data sinks
###

@curried
def showgrid(data, key="input", label=None, grid=(4, 8)):
    """A sink that shows a grid of images.

    :param data: iterator
    :param key: key to be displayed
    :param label: label field to be displayed
    :param grid: grid size for display

    """
    import pylab
    rows, cols = grid
    for i, sample in enumerate(data):
        if i >= rows * cols:
            break
        pylab.subplot(rows, cols, i + 1)
        pylab.xticks([])
        pylab.yticks([])
        pylab.imshow(sample["input"])


@curried
def batchinfo(data, n=1):
    """A sink that provides info about samples/batches.

    :param data: iterator
    :param n: number of samples to print info for

    """
    for i, sample in enumerate(data):
        if i >= n:
            break
        print(type(sample))
        for k, v in list(sample.items()):
            print(k, type(v), end=' ')
            if isinstance(v, np.ndarray):
                print(v.shape, np.amin(v), np.mean(v), np.amax(v), end=' ')
            print()
        print()


def precache(data, cache, key="__key__", max_size=1000000):
    total = 0
    start = None
    cache_full = False
    for sample in data:
        value = objhash(sample[key])
        if start is None:
            start = value
        else:
            if start == value:
                cache_full = True
        if cache_full:
            break
        sample["__index__"] = total
        cache[str(total)] = sample
        sample = dict(sample)
        sample["__epoch__"] = 0
        total += 1
        if total > max_size:
            raise Exception("cache size exceeded")
        yield sample
    if hasattr(cache, "sync") and callable(cache.sync):
        cache.sync()


def cacheserver(cache, start, stop):
    keys = list(cache.keys())
    for epoch in range(start, stop):
        pyr.shuffle(keys)
        for k in keys:
            v = cache[k]
            sample = dict(v)
            sample["__epoch__"] = epoch
            yield sample


@curried
def cached(data, nepochs=1000000, max_size=1000000000, key="__key__"):
    cache = {}
    for sample in precache(data, cache, key=key, max_size=max_size):
        yield sample
    for sample in cacheserver(cache, 1, nepochs):
        yield sample


@curried
def disk_cached(data, nepochs=1000000, max_size=1000000000, key="__key__", path=None):
    try:
        if path is None:
            path = tempfile.mktemp()
        cache = sqlshelve.open(path)
        for sample in precache(data, cache, key=key, max_size=max_size):
            yield sample
        for sample in cacheserver(cache, 1, nepochs):
            yield sample
    finally:
        if os.path.exists(path):
            os.unlink(path)


@curried
def persistent_cached(data, path, nepochs=1000000, max_size=1000000000, key="__key__", verbose=False):
    assert nepochs >= 1
    if not os.path.exists(path):
        temp_path = path+".temp"
        if os.path.exists(temp_path):
            os.unlink(temp_path)
        if verbose:
            print("creating", temp_path)
        cache = sqlshelve.open(temp_path)

        for sample in precache(data, cache, key=key, max_size=max_size):
            yield sample
        os.rename(temp_path, path)
        start = 1
    else:
        assert os.path.exists(path) or os.path.exists(path+".dat")
        if verbose:
            print("opening", path)
        cache = sqlshelve.open(path, protocol=-1)
        start = 0
    for sample in cacheserver(cache, start, nepochs):
        yield sample
    cache.close()
