from __future__ import unicode_literals

import pdb
from builtins import next

import numpy as np
from dlinputs import gopen
import itertools as itt


def test_gopen():
    assert gopen.gopen("/etc/passwd").read().decode()[:4] == "root"


def test_url():
    assert gopen.test_url("/etc/passwd")
    assert not gopen.test_url("/etc/")
    assert not gopen.test_url("/etc/LSKDJFLKJLFJ")
    assert gopen.test_url("http://www.google.com/")
    assert not gopen.test_url("http://www.slkJLKJLFKDJLJ.com/")


def test_find_shards():
    target = "http://storage.googleapis.com/lpr-ocr/uw3-lines.tgz"
    url = gopen.find_shards([
        "http://www.nvidia.com/lpr-ocr/uw3-lines.tgz",
        target
    ])
    assert url == target


def test_sharditerator():
    url = "http://storage.googleapis.com/lpr-ocr/uw3-lines.tgz"
    data = gopen.sharditerator(url)
    for sample in data:
        break
    assert set(["__key__", "png"]) < set(sample.keys()), list(sample.keys())


def test_sharditerator_once():
    url = "http://storage.googleapis.com/lpr-ocr/uw3-lines.tgz"
    data = gopen.sharditerator_once(url)
    for sample in data:
        break
    assert set(["__key__", "png"]) < set(sample.keys()), list(sample.keys())


def test_open_source():
    data = gopen.open_source("testdata/sample.tgz")
    sample = next(data)
    assert isinstance(sample["png"], np.ndarray)

    data = gopen.open_source("testdata/sample.tgz", decode=False)
    sample = next(data)
    assert isinstance(sample["png"], (bytes, str)), sample["png"]

def test_WebLoader():
    loader = gopen.WebLoader("testdata/sample.tgz")
    for sample in loader:
        break
    assert set(["__key__", "png"]) < set(sample.keys()), list(sample.keys())
    assert isinstance(sample["png"], np.ndarray)
    the_key = sample["__key__"]
    for sample in loader:
        break
    assert the_key == sample["__key__"], (the_key, sample["__key__"])

def test_WebLoader2():
    loader = gopen.WebLoader("testdata/sample.tgz", keys="png cls".split())
    for sample in loader:
        break
    assert len(sample) == 2
    assert isinstance(sample[1], int)

def test_WebLoader3():
    loader = gopen.WebLoader("testdata/sample.tgz", keys="png cls".split())
    cls1 = [sample[1] for sample in itt.islice(loader, 0, 10)]
    cls2 = [sample[1] for sample in itt.islice(loader, 0, 10)]
    assert cls1 == cls2, (cls1, cls2)

def test_WebLoader4():
    loader = gopen.WebLoader("testdata/sample.tgz", keys="png cls".split(), epochs=1)
    cls = [sample[1] for sample in loader]
    assert len(cls) == 90
    cls = [sample[1] for sample in loader]
    assert len(cls) == 90
