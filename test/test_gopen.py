from __future__ import unicode_literals

import pdb
from builtins import next
from imp import reload

import numpy as np
from dlinputs import gopen


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
	reload(gopen)
	url = "http://storage.googleapis.com/lpr-ocr/uw3-lines.tgz"
	data = gopen.sharditerator(url)
	for sample in data:
	    break
	assert set(["__key__", "png"]) < set(sample.keys()), list(sample.keys())

def test_sharditerator_once():
	reload(gopen)
	url = "http://storage.googleapis.com/lpr-ocr/uw3-lines.tgz"
	data = gopen.sharditerator_once(url)
	for sample in data:
	    break
	assert set(["__key__", "png"]) < set(sample.keys()), list(sample.keys())

def test_open_source():
	reload(gopen)
	data = gopen.open_source("testdata/sample.tgz")
	sample = next(data)
	assert isinstance(sample["png"], np.ndarray)

	reload(gopen)
	data = gopen.open_source("testdata/sample.tgz", decode=False)
	sample = next(data)
	assert isinstance(sample["png"], (bytes, str)), sample["png"]
