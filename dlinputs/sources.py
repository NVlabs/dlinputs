from __future__ import print_function
from __future__ import absolute_import
# Copyright (c) 2017 NVIDIA CORPORATION. All rights reserved.
# See the LICENSE file for licensing terms (BSD-style).

from builtins import zip
from builtins import range
import os
import re
import glob
import codecs
import random as pyr
import os.path

import numpy as np
import pylab

from . import paths
from . import utils


def infinite(sample):
    """Repeat the same sample over and over again (for testing).

    :param sample: sample to be repeated
    :returns: iterator yielding sample

    """
    while True:
        yield sample

def generator(source, nrepeats=int(1e9)):
    """Repeat data from a source (returned by a callable function).

    :param source: callable function yielding an iterator
    :param nrepeats: number of times to repeat
    :returns: iterator over `nrepeats` repeats of `source`

    """
    for i in range(nrepeats):
        data = source()
        for sample in data:
            yield sample

def check_ds_size(ds, size):
    """Helper function to check the size of a dataset.

    This is mostly just a nice error message

    :param int ds: dataset size
    :param tuple size: lower and upper bounds of dataset size

    """
    if isinstance(size, int): size = (size, size)
    if not isinstance(ds, int): ds = len(ds)
    if ds < size[0]:
        raise ValueError("dataset size is {}, should be in range {}; use size= in dataset iterator"
                         .format(ds, size))
    if ds > size[1]:
        raise ValueError("dataset size is {}, should be in range {}; use size= in dataset iterator"
                         .format(ds, size))

def dirtree(top, extensions, epochs=1,
                shuffle=True, verbose=True, size=(100,1e9),
                decode=True):
    """Iterate of training samples in a directory tree.

    :param top: top of the directory tree
    :param list,str extensions: list/comma separated string of extensions
    :param int epochs: number of epochs to iterate over the data
    :param bool shuffle: whether to shuffle the data
    :param bool verbose: whether to output info about the interator
    :param size: expected dataset size
    :returns: iterator over samples

    """
    if decode is True:
        decode = utils.autodecode
    elif decode is False:
        decode = lambda x: x
    if isinstance(extensions, str):
        extensions = extensions.split(",")
    assert os.path.isdir(top)
    lines = list(paths.find_basenames(top, extensions))
    if verbose: print("got {} samples".format(len(lines)))
    check_ds_size(lines, size)
    for epoch in range(epochs):
        if shuffle: pyr.shuffle(lines)
        for fname in lines:
            result = {}
            result["__path__"] = fname
            result["__epoch__"] = epoch
            for extension in extensions:
                result[extension] = paths.readfile(fname + "." + extension)
            result = decode(result)
            yield result


def basenames(basenamefile, extensions, split=True, epochs=1,
                shuffle=True, verbose=True, size=(100,1e9),
                decode=True):
    """Iterate over training samples given as basenames and extensions.

    :param basenamefile: file containing one basename per line
    :param extensions: list of expected extensions for each basename
    :param split: remove any extension from files in basenamefile
    :param epochs: number of times to iterate
    :param shuffle: shuffle before training
    :param verbose: verbose output
    :param size: expected dataset size
    :returns: iterator

    """
    if decode is True:
        decode = utils.autodecode
    elif decode is False:
        decode = lambda x: x
    if isinstance(extensions, str):
        extensions = extensions.split(",")
    root = os.path.abspath(basenamefile)
    root = os.path.dirname(root)
    with open(basenamefile, "r") as stream:
        lines = [line.strip() for line in stream]
    if verbose: print("got {} samples".format(len(lines)))
    check_ds_size(lines, size)
    for epoch in range(epochs):
        if shuffle: pyr.shuffle(lines)
        for fname in lines:
            if split:
                fname, _ = os.path.splitext(fname)
            result = {}
            path = os.path.join(root, fname)
            result["__path__"] = path
            result["__epoch__"] = epoch
            for extension in extensions:
                result[extension] = paths.readfile(path + "." + extension)
            result = decode(result)
            yield result

def tabular(table, colnames, separator="\t", maxerrors=100, encoding="utf-8",
              epochs=1, shuffle=True, verbose=True, size=(100,1e9), decode=True):
    """Iterate over training samples given by a tabular input.

    Columns whose names start with "_" are passed on directly as strings, all other
    columns are interpreted as file names and read.

    :param str table: tabular input file separated by `separator`
    :param list,str colnames: column names (keys in sample), list or comman separated
    :param str separator: separator for columns in input file
    :param maxerrors: maximum number of read errors
    :param encoding: text file encoding
    :param epochs: number of epochs to iterate
    :param shuffle: shuffle data prior to training
    :param verbose: verbose output
    :param size: expected dataset size
    :returns: iterator

    """
    if decode is True:
        decode = utils.autodecode
    elif decode is False:
        decode = lambda x: x
    if isinstance(size, int):
        size = (size, size)
    if isinstance(colnames, str):
        colnames = colnames.split(",")
    root = os.path.abspath(table)
    root = os.path.dirname(root)
    with codecs.open(table, "r", encoding) as stream:
        lines = stream.readlines()
    if verbose: print("got {} samples".format(len(lines)))
    check_ds_size(lines, size)
    nerrors = 0
    for epoch in range(epochs):
        if shuffle: pyr.shuffle(lines)
        for line in lines:
            line = line.strip()
            if line[0] == "#": continue
            fnames = line.split(separator)
            if len(fnames) != len(colnames):
                print("bad input: {}".format(line))
                if nerrors > maxerrors:
                    raise ValueError("bad input")
                nerrors += 1
                continue
            result = {}
            result["__epoch__"] = epoch
            for name, value in zip(colnames, fnames):
                if name[0]=="_":
                    result[name] = value
                else:
                    path = os.path.join(root, value)
                    if not os.path.exists(path):
                        print("{}: not found".format(path))
                        if nerrors > maxerrors:
                            raise ValueError("not found")
                        nerrors += 1
                        continue
                    result[name] = paths.readfile(path)
                    result["__path__"+name] = path
            result = decode(result)
            yield result


def bookdir(bookdir, epochs=1, shuffle=True):
    """Read a dataset from an OCRopus-style book directory.

    :param bookdir: top level directory in OCRopus bookdir format
    :param epochs: number of epochs to iterate for
    :param shuffle: shuffle the samples prior to reading
    :returns: iterator

    """
    assert os.path.isdir(bookdir), bookdir
    fnames = glob.glob(bookdir + "/????/??????.gt.txt")
    fnames.sort()
    for epoch in range(epochs):
        if shuffle: pyr.shuffle(fnames)
        for fname in fnames:
            base = re.sub(".gt.txt$", "", fname)
            if not os.path.exists(base + ".dew.png"):
                continue
            image = pylab.imread(base + ".dew.png")
            if image.ndim == 3:
                image = np.mean(image, 2)
            image -= np.amin(image)
            image /= np.amax(image)
            image = 1.0 - image
            with codecs.open(fname, "r", "utf-8") as stream:
                transcript = stream.read().strip()
            yield dict(input=image, transcript=transcript, __epoch__=epoch)
