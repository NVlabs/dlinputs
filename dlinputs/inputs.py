# Copyright (c) 2017 NVIDIA CORPORATION. All rights reserved.
# See the LICENSE file for licensing terms (BSD-style).

import codecs
import functools as ft
import glob
import imp
import itertools
import math
import os
import os.path
import random as pyr
import re
import sqlite3
import StringIO
import tarfile
import types
import urllib2
import urlparse
import warnings
from collections import namedtuple

import numpy as np
import pylab
import PIL
import pylab
import scipy.ndimage as ndi
import simplejson
from numpy import cos, sin
import numpy.random as npr

from decorators import itfilter, itmapper, itsink, itsource, prints, ComposableIterator


###
### Helper functions.
###

def find_directory(path, target="", tests=[], verbose=False, error=True):
    """Finds the first instance of target on a path.

    If `tests` is supplied, requires those targets to be present as well.
    This only works if the target is a directory.

    :param list path: colon separated path along which to search or list
    :param str target: target to search for (if empty, finds first path element that exists)
    :param list tests: list of functions of additional predicates
    :param bool verbose: output information about tests performed
    :param bool error: not finding the target is an error
    :returns: path to target
    :rtype: str

    """
    if isinstance(path, str):
        path = path.split(":")
    if isinstance(tests, str):
        tests = [tests]
    for root in path:
        if not os.path.isdir(root):
            continue
        candidate = os.path.join(root, target)
        if verbose: print "trying", candidate
        if not os.path.isdir(candidate):
            continue
        failed = False
        for extra in tests:
            testpath = os.path.join(candidate, extra)
            if verbose: print "testing", testpath
            if not os.path.exists(testpath):
                if verbose: print "FAILED"
                failed = True
                break
        if failed: continue
        return candidate
    if error:
        raise ValueError("cannot find {} on path {}".format(target, path))


def find_file(path, target, tests=[], verbose=False, error=True):
    """Finds the first instance of file on a path.

    :param list path: colon separated path along which to search or list
    :param str target: target to search for (if empty, finds first path element that exists)
    :param list tests: list of functions of additional predicates
    :param bool verbose: output information about tests performed
    :param bool error: not finding the target is an error
    :returns: path to target
    :rtype: str

    """
    if isinstance(path, str):
        path = path.split(":")
    for root in path:
        if not os.path.isdir(root):
            continue
        candidate = os.path.join(root, target)
        if verbose: print "trying", candidate
        if not os.path.isfile(candidate):
            continue
        failed = False
        for extra in tests:
            if not extra(candidate):
                if verbose: print "FAILED"
                failed = True
                break
        if failed: continue
        return candidate
    if error:
        raise ValueError("cannot find {} on path {}".format(target, path))

def readfile(fname):
    """Helper function to read a file (binary).

    :param fname: file name to be read
    :returns: contents of file

    """
    with open(fname, "rb") as stream:
        return stream.read()


def splitallext(path):
    """Helper method that splits off all extension.

    Returns base, allext.

    :param path: path with extensions
    :returns: path with all extensions removed

    """
    match = re.match(r"^((?:.*/|)[^.]+)[.]([^/]*)$", path)
    if not match:
        return None, None
    return match.group(1), match.group(2)



def find_basenames(top, extensions):
    """Finds all basenames that have all the given extensions inside a tree.

    :param top: root of directory tree
    :param extensions: required extensions
    :returns: iterator over basenames

    """
    assert os.path.isdir(top), top
    if isinstance(extensions, str):
        extensions = extensions.split(",")
    if extensions is not None:
        extensions = set(extensions)
    for root, dirs, files in os.walk(top):
        dirs.sort()
        files.sort()
        prefixes = {splitallext(fname)[0] for fname in files}
        for prefix in prefixes:
            missing = [e for e in extensions if prefix+"."+e not in files]
            if len(missing) > 0: continue
            yield os.path.join(root, prefix)


def make_gray(image):
    """Converts any image to a grayscale image by averaging.

    Knows about alpha channels etc.

    :param image: rank 2 or 3 ndarray
    :returns: rank 2 ndarray

    """
    if image.ndim == 2:
        return image
    assert image.ndim == 3
    assert image.shape[2] in [1, 3, 4]
    return np.mean(image[:, :, :3], 2)


def make_rgb(image):
    """Converts any image to an RGB image.

    Knows about alpha channels etc.

    :param image: rank 2 or 3 ndarray
    :returns: rank 3 ndarray of shape :,:,3

    """
    if image.ndim == 2:
        image = image.reshape(image.shape + (1,))
    assert image.ndim == 3
    if image.shape[2] == 1:
        return np.repeat(image, 3, 2)
    elif image.shape[2] == 3:
        return image
    elif image.shape[2] == 4:
        return image[:, :, :3]


def make_rgba(image, alpha=255):
    """Converts any image to an RGBA image.

    Knows about alpha channels etc.

    :param image: rank 2 or 3 ndarray
    :param alpha: default alpha value
    :returns: rank 3 ndarray with shape :,:,4

    """
    if image.ndim == 2:
        image = image.reshape(image.shape + (1,))
    assert image.ndim == 3
    if image.shape[2] == 1:
        result = np.repeat(image, 4, 2)
        result[:, :, 3] = alpha
        return result
    elif image.shape[2] == 3:
        h, w, _ = image.shape
        result = np.zeros((h, w, 4), 'uint8')
        result[:, :, :3] = image
        result[:, :, 3] = alpha
        return result
    elif image.shape[2] == 4:
        return image

def invert_mapping(kvp):
    """Inverts the mapping given by a dictionary.

    :param kvp: mapping to be inverted
    :returns: inverted mapping
    :rtype: dictionary

    """
    return {v: k for k, v in kvp.items()}

def get_string_mapping(kvp):
    """Returns a dictionary mapping strings to strings.

    This can take either a string of the form "name=value:name2=value2"
    or a dictionary containing all string keys and values.

    :param kvp: dictionary or string
    :returns: dictionary

    """
    if kvp is None:
        return {}
    if isinstance(kvp, (str, unicode)):
        return {k: v for k, v in [kv.split("=", 1) for kv in kvp.split(":")]}
    elif isinstance(kvp, dict):
        for k, v in kvp.items():
            assert isinstance(k, str)
            assert isinstance(v, str)
        return kvp
    else:
        raise ValueError("{}: wrong type".format(type(kvp)))


def pilread(stream, color="gray", asfloat=True):
    """Read an image from a stream using PIL.

    :param stream: stream to read the image from
    :param color: "gray", "rgb" or "rgba".
    :param asfloat: return float image instead of uint8 image

    """
    image = PIL.Image.open(stream)
    result = np.array(image, 'uint8')
    if color is None:
        pass
    elif color == "gray":
        result = make_gray(result)
    elif color == "rgb":
        result = make_rgb(result)
    elif color == "rgba":
        result = make_rgba(result)
    else:
        raise ValueError("{}: unknown color space".format(color))
    if asfloat:
        result = result.astype("f") / 255.0
    return result

def pilreads(data, color, asfloat=True):
    """Read an image from a string or buffer using PIL.

    :param data: data to be decoded
    :param color: "gray", "rgb" or "rgba".
    :param asfloat: return float instead of uint8

    """
    assert color is not None
    return pilread(StringIO.StringIO(data), color=color, asfloat=asfloat)


pilgray = ft.partial(pilreads, color="gray")
pilrgb = ft.partial(pilreads, color="rgb")

def iminvert(image):
    """Invert the given image.

    :param image: image
    :returns: inverted image

    """
    assert np.amin(image) >= -1e-6
    assert np.amax(image) <= 1+1e-6
    return 1.0 - clip(image, 0, 1.0)

def autoinvert(image):
    """Autoinvert the given document image.

    If the image appears to be black on white, invert it to white on black,
    otherwise leave it unchanged.

    :param image: document image
    :returns: autoinverted document image

    """
    assert np.amin(image) >= -1e-6
    assert np.amax(image) <= 1+1e-6
    if np.median(image) > 0.5:
        return 1.0 - image
    else:
        return image

def pildumps(image, format="PNG"):
    """Compress an image using PIL and return it as a string.

    Can handle float or uint8 images.

    :param image: ndarray representing an image
    :param format: compression format ("PNG" or "JPEG")

    """
    result = StringIO.StringIO()
    if image.dtype in [np.dtype('f'), np.dtype('d')]:
        assert np.amin(image) > -0.001 and np.amax(image) < 1.001
        image = np.clip(image, 0.0, 1.0)
        image = np.array(image * 255.0, 'uint8')
    PIL.Image.fromarray(image).save(result, format=format)
    return result.getvalue()


pilpng = pildumps
piljpg = ft.partial(pildumps, format="JPEG")


def make_distortions(size, distortions=[(5.0, 3)]):
    """Generate 2D distortions using filtered Gaussian noise.

    The distortions are a sum of gaussian filtered white noise
    with the given sigmas and maximum distortions.

    :param size: size of the image for which distortions are generated
    :param distortions: list of (sigma, maxdist) pairs
    :returns: a grid of source coordinates suitable for scipy.ndimage.map_coordinates

    """
    h, w = size
    total = np.zeros((2, h, w), 'f')
    for sigma, maxdist in distortions:
        deltas = pylab.randn(2, h, w)
        deltas = ndi.gaussian_filter(deltas, (0, sigma, 0))
        deltas = ndi.gaussian_filter(deltas, (0, 0, sigma))
        r = np.amax((deltas[...,0]**2 + deltas[...,1]**2)**.5)
        deltas *= maxdist / r
        total += deltas
    deltas = total
    xy = np.array(np.meshgrid(range(h),range(w))).transpose(0,2,1)
    coords = deltas + xy
    return coords

def map_image_coordinates(image, coords, order=1, mode="nearest"):
    """Given an image, map the image coordinates for each channel.

    :param image: rank 2 or 3 image
    :param coords: coords to map to
    :param order: order of the interpolation
    :param mode: mode for the boundary
    :returns: distorted image

    """
    if image.ndim==2:
        return ndi.map_coordinates(image, coords, order=order)
    elif image.ndim==3:
        result = np.zeros(image.shape, image.dtype)
        for i in range(image.shape[-1]):
            ndi.map_coordinates(image[...,i], coords, order=order, output=result[...,i], mode=mode)
        return result

def random_distortions(images, distortions=[(5.0, 3)], order=1, mode="nearest"):
    """Apply a random distortion to a list of images.

    All images must have the same width and height.

    :param images: list of images
    :param distortions: list of distortion parameters for `make_distortions`
    :param order: order of the interpolation
    :param mode: boundary handling
    :returns: list of distorted images

    """
    h, w = images[0].shape[:2]
    coords = make_distortions((h, w), distortions)
    return [map_image_coordinates(image, coords, order=order, mode=mode) for image in images]

def random_affine(ralpha=(-0.2, 0.2), rscale=((0.8, 1.2), (0.8, 1.2))):
    """Compute a random affine transformation matrix.

    Note that this is random scale and random rotation, not an
    arbitrary affine transformation.

    :param ralpha: range of rotation angles
    :param rscale: range of scales for x and y
    :returns: random affine transformation

    """
    affine = np.eye(2)
    if rscale is not None:
        (x0, x1), (y0, y1) = rscale
        affine = np.diag([npr.uniform(x0, x1), npr.uniform(y0, y1)])
    if ralpha is not None:
        a0, a1 = ralpha
        a = npr.uniform(a0, a1)
        c = cos(a)
        s = sin(a)
        m = np.array([[c, -s], [s, c]], 'f')
        affine = np.dot(m, affine)
    return affine


def random_gamma(image, rgamma=(0.5, 2.0), cgamma=(0.8, 1.2)):
    """Perform a random gamma transformation on an image.

    :param image: input image
    :param rgamma: grayscale gamma range
    :param cgamma: separate per channel color gamma range
    :returns: transformed image

    """
    image = image.copy()
    if rgamma is not None:
        gamma = npr.uniform(*rgamma)
    else:
        gamma = 1.0
    for plane in range(3):
        g = gamma
        if cgamma is not None:
            g *= npr.uniform(*cgamma)
        image[..., plane] = image[..., plane] ** g
    return image


def standardize(image, size, crop=0, mode="nearest", affine=np.eye(2)):
    """Rescale and crop the image to the given size.

    With crop=0, this rescales the image so that the target size fits
    snugly into it and cuts out the center; with crop=1, this rescales
    the image so that the image fits into the target size and fills
    the boundary in according to `mode`.

    :param ndarray image: image to be standardized
    :param tuple size: target size
    :param bool crop: crop the image
    :param str mode: boundary mode
    :param affine: affine transformation to be applied
    :returns: standardized image
    :rtype: ndarray

    """
    h, w = image.shape[:2]
    th, tw = size
    oshape = (th, tw, image.shape[2])
    if crop:
        scale = min(h * 1.0 / th, w * 1.0 / tw)
    else:
        scale = max(h * 1.0 / th, w * 1.0 / tw)
    affine = np.eye(2)
    affine = affine * scale
    center = np.array(image.shape[:2], 'f') / 2
    tcenter = np.array([th, tw], 'f') / 2
    delta = np.matmul(affine, tcenter) - center
    matrix = np.eye(3)
    matrix[:2, :2] = affine
    offset = np.zeros(3)
    offset[:2] = -delta
    result = ndi.affine_transform(image, matrix, offset, order=1,
                                  output_shape=oshape, mode=mode)
    return result

def samples_to_batch(samples, tensors=True):
    """Take a collection of samples (dictionaries) and create a batch.

    If `tensors` is True, `ndarray` objects are combined into
    tensor batches.

    :param dict samples: list of samples
    :param bool tensors: whether to turn lists of ndarrays into a single ndarray
    :returns: single sample consisting of a batch
    :rtype: dict

    """
    result = {k: [] for k in samples[0].keys()}
    for i in range(len(samples)):
        for k in result.keys():
            result[k].append(samples[i][k])
    if tensors == True:
        tensors = [x for x in result.keys()
                   if isinstance(result[x][0], (np.ndarray, int, float))]
    for k in tensors:
        result[k] = np.array(result[k])
    return result


def intlist_to_hotonelist(cs, nc, allow_bad_classes=True):
    """Helper function for LSTM/CTC-based OCR: encode ground truth as array.

    Given a list of target classes `cs` and a total
    maximum number of classes, compute an array that has
    a `1` in each column and time step corresponding to the
    target class, with class 0 interspersed.

    :param cs: list of target classes
    :param nc: total number of classes
    :returns: ndarray representing a hotone encoding

    """
    result = np.zeros((2*len(cs)+1,nc))
    for i,j in enumerate(cs):
        result[2*i,0] = 1.0
        if allow_bad_classes:
            j = min(j, nc-1) # FIX for bad inputs
        result[2*i+1,j] = 1.0
    result[-1,0] = 1.0
    return result

def hotonelist_to_intlist(outputs,threshold=0.7,pos=0):
    """Helper function for LSTM-based OCR: decode LSTM outputs.

    Translate back. Thresholds on class 0, then assigns the maximum class to
    each region. ``pos`` determines the depth of character information returned:
    - `pos=0`: Return list of recognized characters
    - `pos=1`: Return list of position-character tuples
    - `pos=2`: Return list of character-probability tuples


    :param outputs: 2D array containing posterior probabilities
    :param threshold: posterior probability threshold
    :param pos: what to return
    :returns: decoded hot one outputs

    """
    labels,n = measurements.label(outputs[:,0]<threshold)
    mask = np.tile(labels.reshape(-1,1),(1,outputs.shape[1]))
    maxima = measurements.maximum_position(outputs,mask,np.arange(1,np.amax(mask)+1))
    if pos==1: return maxima # include character position
    if pos==2: return [(c, outputs[r,c]) for (r,c) in maxima] # include character probabilities
    return [c for (r,c) in maxima] # only recognized characters


def spliturl(url):
    """Split a URL into its extension and base.

    :param url: input url
    :returns: tuple of basename and extension

    """
    match = re.search(r"^(.+)\.([^:/]+)$", url)
    if match:
        return match.group(1), match.group(2)
    else:
        return url, ""


def read_url_path(url, urlpath, verbose=False):
    """Attempts to find `url` on the `urlpath`.

    :param url: relative URL
    :param urlpath: list or space separated string of base URLs
    :param verbose: inform user about trials
    :returns: contents first URL that can be opened, base url
    :rtype: tuple

    """
    if isinstance(urlpath, str):
        urlpath = urlpath.strip().split()
    if urlpath is None:
        urlpath = [re.sub("[^/]+$", "", url)]
    for base in urlpath:
        trial = urlparse.urljoin(base, url)
        if verbose: print "trying: {}".format(trial)
        try:
            return openurl(trial).read(), base
        except urllib2.URLError:
            if verbose: print trial, ": FAILED"
            continue
    return None

url_rewriter = None

def findurl(url):
    """Finds a URL using environment variables for helpers.

    If DLP_URLREWRITER is set in the environment, it is loaded.
    If dlinputs. url_rewriter is not None, it is applied to the url.
    If DLP_URLBASE is not None, it is joined to the url as a base url.

    FIXME: make DLP_URLBASE a path

    :param url: url to locate
    :return: located URL
    :rtype: str

    """
    rewriter = os.environ.get("DLP_URLREWRITER", None)
    if rewriter is not None:
        execfile(rewriter)
    if url_rewriter is not None:
        url = url_rewriter(url)
    base = os.environ.get("DLP_URLBASE", None)
    if base is not None:
        url = urlparse.urljoin(base, url)
    return url


def openurl(url):
    """Open a URL with findurl followed by urlopen.

    :param url: url to be opened
    :returns: result of urlopen
    :rtype: stream

    """
    url = findurl(url)
    return urllib2.urlopen(url)


def find_url(paths, extra=None):
    """Given a list of url paths, find the first one that matches.

    :param paths: list of base urls
    :param extra: extra relative URL to be joined before testing
    :returns: first path that succeeds
    """

    if isinstance(paths, str):
        paths = paths.split()
    for path in paths:
        test = path
        if extra is not None:
            test = urllib2.urljoin(path, extra)
        try:
            urllib2.urlopen(test)
            return path
        except:
            pass
    return None


def read_shards(url, shardtype="application/x-tgz", urlpath=None, verbose=True):
    """Read a shards description file from a URL and convert relative URLs in shard file.

    :param url: url to read shard file from (JSON format)
    :param shardtype: default shard type
    :param urlpath: path on which to search for the shard file
    :param verbose: output progress
    :returns: list of URLs for shard

    """
    data, base = read_url_path(url, urlpath, verbose=verbose)
    if verbose:
        print "# read_shards", url, "base", base
    if data is None:
        raise Exception("url not found") # FIXME
    shards = simplejson.loads(data)
    if shards is None:
        raise Exception("cannot find {} on {}".format(url, urlpath))
    if shardtype is not None and "shardtype" in shards:
        assert shards["shardtype"] == "application/x-tgz", shards["shardtype"]
    shards = shards["shards"]
    for s in shards:
        for i in range(len(s)):
            s[i] = urlparse.urljoin(base, s[i])
    return shards

def extract_shards(url):
    """Extract a shard list from a shard URL.

    Shard URLs are URLs containing a string of the form `@000123`.
    This denotes that the shards are given by a six digit string and
    that there are 123 shards.

    :param url: shard url
    :returns: list of URLs for shard

    """
    prefix, shards, suffix = re.search(r"^(.*)(@[0-9]+)(.*)$", url).groups()
    f = len(shards) - 1
    n = int(shards[1:])
    result = []
    for i in xrange(n):
        result.append("%s%0*d%s" % (prefix, f, i, suffix))
    return result

ShardEntry = namedtuple("ShardEntry", "name,prefix,index,suffix")

def iterate_shards(url):
    """Iterates over shards.

    Given a string of the form "prefix-@000123-suffix", yields
    objects containing:

    - obj.name: expanded name
    - obj.prefix: prefix value
    - obj.index: index value (as zero-padded string)
    - obj.suffix: suffix value

    :param url: shard url
    :returns: iterator over shards

    """
    prefix, shards, suffix = re.search(r"^(.*)(@[0-9]+)(.*)$", url).groups()
    f = len(shards) - 1
    n = int(shards[1:])
    result = []
    for i in xrange(n):
        index = "%0*d" % (f, i)
        yield ShardEntry(prefix+index+suffix, prefix, index, suffix)

###
### Data sources.
###


@itsource
def itinfinite(sample):
    """Repeat the same sample over and over again (for testing).

    :param sample: sample to be repeated
    :returns: iterator yielding sample

    """
    while True:
        yield sample


@itsource
def itrepeat(source, nrepeats=int(1e9)):
    """Repeat data from a source (returned by a callable function).

    :param source: callable function yielding an iterator
    :param nrepeats: number of times to repeat
    :returns: iterator over `nrepeats` repeats of `source`

    """
    for i in xrange(nrepeats):
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

@itsource
def itdirtree(top, extensions, epochs=1,
                shuffle=True, verbose=True, size=(100,1e9)):
    """Iterate of training samples in a directory tree.

    :param top: top of the directory tree
    :param list,str extensions: list/comma separated string of extensions
    :param int epochs: number of epochs to iterate over the data
    :param bool shuffle: whether to shuffle the data
    :param bool verbose: whether to output info about the interator
    :param size: expected dataset size
    :returns: iterator over samples

    """
    if isinstance(extensions, str):
        extensions = extensions.split(",")
    assert os.path.isdir(top)
    lines = list(find_basenames(top, extensions))
    if verbose: print "got {} samples".format(len(lines))
    check_ds_size(lines, size)
    for epoch in xrange(epochs):
        if shuffle: pyr.shuffle(lines)
        for fname in lines:
            result = {}
            result["__path__"] = fname
            result["__epoch__"] = epoch
            for extension in extensions:
                result[extension] = readfile(fname + "." + extension)
            yield result


@itsource
def itbasenames(basenamefile, extensions, split=True, epochs=1,
                shuffle=True, verbose=True, size=(100,1e9)):
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
    if isinstance(extensions, str):
        extensions = extensions.split(",")
    root = os.path.abspath(basenamefile)
    root = os.path.dirname(root)
    with open(basenamefile, "r") as stream:
        lines = [line.strip() for line in stream.xreadlines()]
    if verbose: print "got {} samples".format(len(lines))
    check_ds_size(lines, size)
    for epoch in xrange(epochs):
        if shuffle: pyr.shuffle(lines)
        for fname in lines:
            if split:
                fname, _ = os.path.splitext(fname)
            result = {}
            path = os.path.join(root, fname)
            result["__path__"] = path
            result["__epoch__"] = epoch
            for extension in extensions:
                result[extension] = readfile(path + "." + extension)
            yield result

@itsource
def ittabular(table, colnames, separator="\t", maxerrors=100, encoding="utf-8",
              epochs=1, shuffle=True, verbose=True, size=(100,1e9)):
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
    if isinstance(size, int):
        size = (size, size)
    if isinstance(colnames, str):
        colnames = colnames.split(",")
    root = os.path.abspath(table)
    root = os.path.dirname(root)
    with codecs.open(table, "r", encoding) as stream:
        lines = stream.readlines()
    if verbose: print "got {} samples".format(len(lines))
    check_ds_size(lines, size)
    for epoch in xrange(epochs):
        if shuffle: pyr.shuffle(lines)
        for line in lines:
            line = line.strip()
            if line[0] == "#": contine
            fnames = line.split(separator)
            if len(fnames) != len(colnames):
                print "bad input: {}".format(line)
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
                        print "{}: not found".format(path)
                        if nerrors > maxerrors:
                            raise ValueError("not found")
                        nerrors += 1
                        continue
                    result[name] = readfile(path)
                    result["__path__"+name] = path
            yield result


def base_plus_ext(fname):
    """Splits pathnames into the file basename plus the extension."""
    return splitallext(fname)

def dir_plus_file(fname):
    """Splits pathnames into the dirname plus the filename."""
    return os.path.split(fname)

def last_dir(fname):
    """Splits pathnames into the last dir plus the filename."""
    dirname, plain = os.path.split(fname)
    prefix, last = os.path.split(dirname)
    return last, plain


def ittarreader1(archive, check_sorted=True, keys=base_plus_ext):
    """Read samples from a tar archive, either locally or given by URL.

    Tar archives are assumed to be sorted by file name. For each basename,
    reads all the files with different extensions and returns a dictionary
    with the extension as key and the file contents as value.

    :param str archive: tar archive with sorted file names (file name or URL)
    :param bool check_sorted: verify that file names are sorted
    :returns: iterator over samples

    """
    if isinstance(archive, str):
        if re.match(r"^(https?|file|s?ftp):(?i)", archive):
            archive = urllib2.urlopen(archive)
        elif re.match(r"^gs:(?i)", archive):
            archive = os.popen("gsutil cat '%s'" % archive, "rb")
    current_count = 0
    current_prefix = None
    current_sample = None
    if isinstance(archive, str):
        stream = tarfile.open(archive, mode="r:*")
    else:
        stream = tarfile.open(fileobj=archive, mode="r|*")
    for tarinfo in stream:
        if not tarinfo.isreg():
            continue
        fname = tarinfo.name
        if fname is None:
            warnings.warn("tarinfo.name is None")
            continue
        prefix, suffix = keys(fname)
        if prefix is None:
            warnings.warn("prefix is None for: %s" % (tarinfo.name,))
            continue
        if prefix != current_prefix:
            if check_sorted and prefix <= current_prefix:
                raise ValueError("[%s] -> [%s]: tar file does not contain sorted keys" % \
                                 (current_prefix, prefix))
            if current_sample is not None and \
               not current_sample.get("__bad__", False):
                yield current_sample
            current_prefix = prefix
            current_sample = dict(__key__=prefix)
        try:
            data = stream.extractfile(tarinfo).read()
        except tarfile.ReadError, e:
            print "tarfile.ReadError at", current_count
            print "file:", tarinfo.name
            print e
            current_sample["__bad__"] = True
        else:
            current_sample[suffix] = data
            current_count += 1
    if len(current_sample.keys()) > 0:
        yield current_sample
    try: del stream
    except: pass
    try: del archive
    except: pass


@itsource
def ittarreader(archive, epochs=1, **kw):
    for epoch in xrange(epochs):
        source = ittarreader1(archive, **kw)
        for sample in source:
            sample["__epoch__"] = epoch
            yield sample

@itsource
def ittarshards(url, shardtype="application/x-tgz", randomize=True, epochs=1,
                urlpath=None, verbose=True):
    """Read a sharded data set, using a JSON-format shards file to find the shards.

    :param url: URL for the shard file (JSON format)
    :param shardtype: the file type for the shards
    :param randomize: shuffle the shards prior to reading
    :param epochs: number of epochs to train for
    :param urlpath: path of base URLs to search for for url
    :param verbose: print info about what is being read

    """
    epochs = int(epochs)
    if url.endswith(".shards"):
        shards = read_shards(url, shardtype=shardtype, urlpath=urlpath,
                             verbose=verbose)
    else:
        shards = extract_shards(url)
        shards = [[s] for s in shards]
    assert isinstance(shards, list)
    assert isinstance(shards[0], list)
    for epoch in xrange(epochs):
        l = list(shards)
        if randomize:
            pyr.shuffle(l)
        for s in l:
            u = pyr.choice(s)
            if verbose:
                print "# reading", s
            try:
                for item in ittarreader(u):
                    item["__shard__"] = u
                    item["__epoch__"] = epoch
                    yield item
            except tarfile.ReadError:
                print "read error in:", u

@itsource
def itsqlite(dbfile, table="train", epochs=1, cols="*", extra="", verbose=False):
    """Read a dataset from an sqlite3 dbfile and the given table.

    Returns samples as dictionaries, with column names as keys
    and column contents as values (values as returned by sqlite3).

    :param dbfile: SQLite database file
    :param table: table name to iterate over
    :param epochs: number of epochs to iterate for
    :param cols: columns to extract in the sample
    :param extra: extra clause for SQL query statement (e.g., "order by rand(), "limit 100")
    :param verbose: output more detailed information
    :returns: iterator over samples

    """
    assert "," not in table
    if "::" in dbfile:
        dbfile, table = dbfile.rsplit("::", 1)
    assert os.path.exists(dbfile)
    sql = "select %s from %s %s" % (cols, table, extra)
    if verbose:
        print "#", sql
    for epoch in xrange(epochs):
        if verbose:
            print "# epoch", epoch, "dbfile", dbfile
        db = sqlite3.connect(dbfile)
        c = db.cursor()
        for row in c.execute(sql):
            cols = [x[0] for x in c.description]
            sample = {k: v for k, v in zip(cols, row)}
            sample["__epoch__"] = epoch
            yield sample
        c.close()
        db.close()


@itsource
def itbookdir(bookdir, epochs=1, shuffle=True):
    """Read a dataset from an OCRopus-style book directory.

    :param bookdir: top level directory in OCRopus bookdir format
    :param epochs: number of epochs to iterate for
    :param shuffle: shuffle the samples prior to reading
    :returns: iterator

    """
    assert os.path.isdir(bookdir), bookdir
    fnames = glob.glob(bookdir + "/????/??????.gt.txt")
    fnames.sort()
    for epoch in xrange(epochs):
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


@itsource
def itmerge(sources, weights=None):
    """Merge samples from multiple sources into a single iterator.

    :param sources: list of iterators
    :param weights: weights for sampling
    :returns: iterator

    """
    source = list(sources)
    assert weights is None, "weighted sampling not implemented yet"
    while len(sources) > 0:
        index = pyr.randint(0, len(sources)-1)
        try:
            sample = sources[index].next()
        except StopIteration:
            del sources[index]
        yield sample
    raise StopIteration()

@itsource
def itconcat(sources, maxepoch=1):
    """Concatenate multiple sources, usually for test sets.

    :param sources: list of iterators
    :param maxepochs: number of epochs (default=1)
    :returns: iterator

    """
    count = 0
    for source in sources:
        for sample in source:
            if maxepoch is not None:
                if sample["__epoch__"] >= maxepoch:
                    break
            sample = dict(sample)
            sample["__count__"] = count
            yield sample
            count += 1

###
### Basic Filters
###

def print_sample(sample):
    """Pretty print a standard sample.

    :param dict sample: key value pairs used for training

    """
    for k in sorted(sample.keys()):
        v = sample[k]
        print k,
        if isinstance(v, np.ndarray):
            print v.dtype, v.shape
        elif isinstance(v, (str, unicode)):
            print repr(v)[:60]
        elif isinstance(v, (int, float)):
            print v
        elif isinstance(v, buffer):
            print type(v), len(v)
        else:
            print type(v), repr(v)[:60]

@itfilter
def itinfo(data, every=0):
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
            print "# itinfo", count
            print_sample(sample)
        count += 1
        yield sample


@itfilter
def itgrep(source, **kw):
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
            for k, v in kw.items():
                matching = not not re.search(v, data[k])
                if matching == _not:
                    skip = True
                    break
            if skip:
                continue
            yield data


@itfilter
def itselect(source, **kw):
    """Select samples from the source that match given predicates.

    Arguments are of the form `fieldname=predicate`.

    :param source: iterator
    :param kw: fieldname=predicate selectors
    :returns: iterator

    """
    for item in source:
        for data in source:
            skip = False
            for k, f in kw.items():
                matching = not not f(data[k])
                if not matching:
                    skip = True
                    break
            if skip:
                continue
            yield data


@itfilter
def itren(data, keep_all=False, keep_meta=True, skip_missing=False, **kw):
    """Rename and select fields using new_name="old_name" keyword arguments.

    :param data: iterator
    :param keep_all: keep all fields, even those that haven't been renamed
    :param keep_meta: keep metadata (fields starting with "_")
    :param skip_missing: skip records where not all fields are present
    :param kw: new_name="old_name" rename rules
    :returns: iterator

    """
    assert not keep_all
    for sample in data:
        skip = False
        result = {}
        if keep_meta:
            for k, v in sample.items():
                if k[0]=="_":
                    result[k] = v
        for k, v in kw.items():
            if v not in sample:
                skip = True
                break
            result[k] = sample[v]
        if skip and skip_missing:
            if skip_missing is 1:
                print v, ": missing field; skipping"
                print_sample(sample)
            continue
        yield result

@itfilter
def itcopy(data, **kw):
    """Copy fields.

    :param data: iterator
    :param kw: new_value="old_value"
    :returns: iterator

    """
    for sample in data:
        result = {k: v for k, v in sample.items()}
        for k, v in kw.items():
            result[k] = result[v]
        yield result

@itfilter
def itmap(data, **keys):
    """Map the fields in each sample using name=function arguments.

    Unmentioned fields are left alone.

    :param data: iterator
    :param keys: name=function pairs, applying function to each sample field
    :returns: iterator

    """
    for sample in data:
        sample = sample.copy()
        for k, f in keys.items():
            sample[k] = f(sample[k])
        yield sample

@itfilter
def ittransform(data, f=None):
    """Map entire samples using the given function.

    :param data: iterator
    :param f: function from samples to samples
    :returns: iterator over transformed samples

    """

    if f is None: f = lambda x: x
    for sample in data:
        yield f(sample)

@itfilter
def itshuffle(data, bufsize=1000):
    """Shuffle the data in the stream.

    This uses a buffer of size `bufsize`. Shuffling at
    startup is less random; this is traded off against
    yielding samples quickly.

    :param data: iterator
    :param bufsize: buffer size for shuffling
    :returns: iterator

    """
    buf = []
    for sample in data:
        if len(buf) < bufsize:
            buf.append(data.next())
        k = pyr.randint(0, len(buf) - 1)
        sample, buf[k] = buf[k], sample
        yield sample


@itfilter
def itbatch(data, batchsize=20, tensors=True, partial=True):
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
            yield samples_to_batch(batch, tensors=tensors)
            batch = []
        batch.append(sample)
    if len(batch) == 0:
        return
    elif len(batch) == batchsize or partial:
        yield samples_to_batch(batch, tensors=tensors)


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

@itfilter
def itunbatch(data):
    """Unbatch data.

    :param data: iterator over batches
    :returns: iterator over samples

    """
    for sample in data:
        keys = sample.keys()
        bs = len(sample[keys[0]])
        for i in xrange(bs):
            yield {k: maybe_index(v, i) for k, v in sample.items()}

@itfilter
def itslice(*args):
    """A pipable version of itertools.islice.

    (Same arguments as islice)

    """
    return itertools.islice(*args)

@itmapper
def itdistort(sample, distortions=[(5.0, 5)], keys=["image"]):
    """Apply distortions to images in sample.

    :param sample
    :param distortions: distortion parameters
    :param keys: fields to distort
    :returns: distorted sample

    """
    images = [sample[k] for k in keys]
    distorted = random_distortions(images, distortions)
    result = dict(sample)
    for k, v in zip(keys, distorted): result[k] = v
    return result

@itmapper
def itstandardize(sample, size, keys=["image"], crop=0, mode="nearest",
                  ralpha=None, rscale=((0.8, 1.0), (0.8, 1.0)),
                  rgamma=None, cgamma=(0.8, 1.2)):
    """Standardize images in a sample.

    :param sample: sample
    :param size: target size
    :param keys: keys for images to be distorted
    :param crop: whether to crop
    :param mode: boundary mode
    :param ralpha: random rotation range (no affine if None)
    :param rscale: random scale range
    :param rgamma: random gamma range (no color distortion if None)
    :param cgamma: random color gamma range
    :returns: standardized szmple

    """
    if ralpha is True: ralpha = (-0.2, 0.2)
    if rgamma is True: rgamma = (0.5, 2.0)
    if ralpha is not None:
        affine = random_affine(ralpha=ralpha, rscale=rscale)
    else:
        affine = np.eye(2)
    for key in keys:
        sample[key] = standardize(
            sample[key], size, crop=crop, mode=mode, affine=affine)
    if rgamma is not None:
        for key in keys:
            sample[key] = random_gamma(sample[key],
                                       rgamma=rgamma,
                                       cgamma=cgamma)
    return sample

###
### Specialized input pipelines for OCR, speech, and related tasks.
###

def ld_makeseq(image):
    """Turn an image into an LD sequence.

    :param image: input image
    :returns: LD sequence

    """
    assert isinstance(image, np.ndarray), type(image)
    if image.ndim==3 and image.shape[2]==3:
        image = np.mean(image, 2)
    elif image.ndim==3 and image.shape[2]==1:
        image = image[:,:,0]
    assert image.ndim==2
    return image.T

def seq_makebatch(images, for_target=False):
    """Given a list of LD sequences, make a BLD batch tensor.

    This performs zero padding as necessary.

    :param images: list of images as LD sequences
    :param for_target: require ndim==2, inserts training blank steps.
    :returns: batched image sequences

    """
    assert isinstance(images, list), type(images)
    assert isinstance(images[0], np.ndarray), images
    if images[0].ndim==2:
        l, d = np.amax(np.array([img.shape for img in images], 'i'), axis=0)
        ibatch = np.zeros([len(images), int(l), int(d)])
        if for_target:
            ibatch[:, :, 0] = 1.0
        for i, image in enumerate(images):
            l, d = image.shape
            ibatch[i, :l, :d] = image
        return ibatch
    elif images[0].ndim==3:
        assert not for_target
        h, w, d = np.amax(np.array([img.shape for img in images], 'i'), axis=0)
        ibatch = np.zeros([len(images), h, w, d])
        for i, image in enumerate(images):
            h, w, d = image.shape
            ibatch[i, :h, :w, :d] = image
        return ibatch

def images2seqbatch(images):
    """Given a list of images, return a BLD batch tensor.

    :param images: list of images
    :returns: ndarray representing batch

    """
    images = [ld_makeseq(x) for x in images]
    return seq_makebatch(images)

def images2batch(images):
    """Given a list of images, return a batch tensor.

    :param images: list of imags
    :returns: batch tensor

    """
    return seq_makebatch(images)

class AsciiCodec(object):
    """An example of a codec, used for turning strings into tensors."""
    def _encode_char(self, c):
        if c=="": return 0
        return max(1, ord(c) - ord(" ") + 1)
    def _decode_char(self, c):
        if c==0: return ""
        return chr(ord(" ") + c - 1)
    def size(self):
        """The number of classes. Zero is always reserved for the empty class.
        """
        return 97
    def encode(self, s):
        """Encode a string.

        :param s: string to be encoded
        :returns: list of integers

        """
        return [self._encode_char(c) for c in s]
    def decode(self, l):
        """Decode a numerical encoding of a string.

        :param l: list of integers
        :returns: string

        """
        return "".join([self._decode_char(x) for x in l])

ascii_codec = AsciiCodec()

def maketarget(s, codec=ascii_codec):
    """Turn a string into an LD target.

    :param s: string
    :param codec: codec
    :returns: hot one encoding of string

    """
    assert isinstance(s, (str, unicode)), (type(s), s)
    codes = codec.encode(s)
    n = codec.size()
    return intlist_to_hotonelist(codes, n)

def transcripts2batch(transcripts, codec=ascii_codec):
    """Given a list of strings, makes ndarray target arrays.

    :param transcripts: list of strings
    :param codec: encoding codec
    :returns: batched hot one encoding of strings suitable for CTC

    """
    targets = [maketarget(s, codec=codec) for s in transcripts]
    return seq_makebatch(targets, for_target=True)

@itfilter
def itbatchedbuckets(data, batchsize=5, scale=1.8, seqkey="input", batchdim=1):
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
        r = int(math.floor(math.log(l) / math.log(scale)))
        batched = buckets.get(r, {})
        for k, v in sample.items():
            if k in batched:
                batched[k].append(v)
            else:
                batched[k] = [v]
        if len(batched[seqkey]) >= batchsize:
            batched["_bucket"] = r
            yield batched
            batched = {}
        buckets[r] = batched
    for r, batched in buckets.items():
        if batched == {}: continue
        batched["_bucket"] = r
        yield batched

@itfilter
def itlineseqbatcher(data, input="image", transcript="transcript", target="target", codec=ascii_codec):
    """Performs text line batching for OCR.

    Usually this is used after itbatchedbuckets.

    :param data: iterator over OCR training samples
    :param input: input field name
    :param transcript: transcript field name
    :param target: target field name
    :param codec: codec used for encoding classes
    :returns: batched sequences

    """
    for sample in data:
        sample = sample.copy()
        sample[input] = images2batch(sample[input])
        sample[target] = transcripts2batch(sample[transcript], codec=codec)
        yield sample

@itfilter
def itlinebatcher(data, input="input", transcript="transcript", target="target", codec=ascii_codec):
    """Performs text line batching for OCR.

    Usually this is used after itbatchedbuckets.

    :param data: iterator over OCR training samples
    :param input: input field name
    :param transcript: transcript field name
    :param target: target field name
    :param codec: codec used for encoding classes
    :returns: batched sequences

    """
    for sample in data:
        sample = sample.copy()
        sample[input] = images2batch(sample[input])
        sample[target] = transcripts2batch(sample[transcript], codec=codec)
        yield sample

###
### various data sinks
###

@itsink
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


@itsink
def batchinfo(data, n=1):
    """A sink that provides info about samples/batches.

    :param data: iterator
    :param n: number of samples to print info for

    """
    for i, sample in enumerate(data):
        if i >= n:
            break
        print type(sample)
        for k, v in sample.items():
            print k, type(v),
            if isinstance(v, np.ndarray):
                print v.shape, np.amin(v), np.mean(v), np.amax(v),
            print
        print


gen = ComposableIterator
