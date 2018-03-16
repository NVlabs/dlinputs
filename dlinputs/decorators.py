# Copyright (c) 2017 NVIDIA CORPORATION. All rights reserved.
# See the LICENSE file for licensing terms (BSD-style).

import itertools
from functools import wraps

verbose = True


def iterable(x):
    return hasattr(x, "next") and hasattr(x, "__iter__")


class ComposableIterator(object):
    def __init__(self, it):
        self.it = it

    def __iter__(self):
        return self

    def __or__(self, f):
        if hasattr(f, "__immediate__"):
            return f.__immediate__(self)
        result = f(self.it)
        assert iterable(result), "{}: did not yield an iterator".format(f)
        result = ComposableIterator(result)
        return result

    def next(self):
        return self.it.next()


class prints(object):
    def __init__(self, n=20):
        self.n = n

    def __immediate__(self, it):
        for i, x in enumerate(itertools.islice(it, self.n)):
            if isinstance(x, dict):
                print i, x.keys()
            else:
                print i, x


def itsource(f):
    @wraps(f)
    def wrapper(*args, **kw):
        return ComposableIterator(f(*args, **kw))
    return wrapper


def itfilter(f):
    @wraps(f)
    def wrapper(*args, **kw):
        def wrapped(iter):
            for sample in f(iter, *args, **kw):
                yield sample
        return wrapped
    return wrapper


def itmapper(f):
    @wraps(f)
    def wrapper(*args, **kw):
        def wrapped(iter):
            for sample in iter:
                yield f(sample, *args, **kw)
        return wrapped
    return wrapper


def itsink(f):
    @wraps(f)
    def wrapper(*args, **kw):
        def wrapped(iter):
            f(iter, *args, **kw)
            return (x for x in [])
        return wrapped
    return wrapper


@itsource
def source0(n=15):
    for i in xrange(n):
        yield i


@itmapper
def test1(x, y):
    print "test1"
    return (y, x)


@itfilter
def test2(iter, y):
    print "test2"
    for sample in iter:
        print "iter"
        yield (sample, y)
