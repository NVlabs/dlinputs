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

import numpy as np
import PIL
import pylab
import scipy.ndimage as ndi
import simplejson
from numpy import cos, sin
from numpy.random import uniform

from decorators import itfilter, itmapper, itsink, itsource, prints, ComposableIterator


###
### Helper functions.
###

def find_directory(path, target="", tests=[], verbose=False, error=True):
    """Finds the first instance of target on a path.

    If `tests` is supplied, requires those targets to be present as well.
    This only works if the target is a directory.
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
    """Helper function to read a file (binary)."""
    with open(fname, "rb") as stream:
        return stream.read()


def splitallext(path):
    """Helper method that splits off all extension.

    Returns base, allext.
    """
    match = re.match(r"^((?:.*/|)[^.]+)[.]([^/]*)$", path)
    if not match:
        return None, None
    return match.group(1), match.group(2)



def find_basenames(top, extensions):
    """Finds all basenames that have all the given extensions inside a tree."""
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
    """
    if image.ndim == 2:
        return image
    assert image.ndim == 3
    assert image.shape[2] in [1, 3, 4]
    return np.mean(image[:, :, :3], 2)


def make_rgb(image):
    """Converts any image to an RGB image.

    Knows about alpha channels etc.
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
        result = zeros((h, w, 4), 'uint8')
        result[:, :, :3] = image
        result[:, :, 3] = alpha
        return result
    elif image.shape[2] == 4:
        return image

def invert_mapping(kvp):
    return {v: k for k, v in kvp.items()}

def get_string_mapping(kvp):
    """Returns a dictionary mapping strings to strings.

    This can take either a string of the form "name=value:name2=value2"
    or a dictionary containing all string keys and values.
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

    Returns a uint8 image if asfloat=False,
    otherwise a float image with values in [0,1].
    Color can be "gray", "rgb" or "rgba".
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

def pilreads(data, color="gray", asfloat=True):
    """Read an image from a string or buffer using PIL.

    Returns a uint8 image if asfloat=False,
    otherwise a float image with values in [0,1].
    Color can be "gray", "rgb" or "rgba".
    """
    return pilread(StringIO.StringIO(data), color=color, asfloat=asfloat)


pilgray = ft.partial(pilreads, color="gray")
pilrgb = ft.partial(pilreads, color="rgb")


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


pilpng = pildumps
piljpg = ft.partial(pildumps, format="JPEG")


def random_affine(ralpha=(-0.2, 0.2), rscale=((0.8, 1.0), (0.8, 1.0))):
    """Compute a random affine transformation matrix."""
    affine = np.eye(2)
    if rscale is not None:
        (x0, x1), (y0, y1) = rscale
        affine = np.diag([uniform(x0, x1), uniform(y0, y1)])
    if ralpha is not None:
        a0, a1 = ralpha
        a = uniform(a0, a1)
        c = cos(a)
        s = sin(a)
        m = np.array([[c, -s], [s, c]], 'f')
        affine = np.dot(m, affine)
    return affine


def random_gamma(image, rgamma=(0.5, 2.0), cgamma=(0.8, 1.2)):
    """Perform a random gamma transformation on an image."""
    image = image.copy()
    if rgamma is not None:
        gamma = uniform(*rgamma)
    else:
        gamma = 1.0
    for plane in range(3):
        g = gamma
        if cgamma is not None:
            g *= uniform(*cgamma)
        image[..., plane] = image[..., plane] ** g
    return image


def standardize(image, size, crop=0, mode="nearest", affine=np.eye(2)):
    """Rescale and crop the image to the given size. With crop=0,
    this rescales the image so that the target size fits snugly
    into it and cuts out the center; with crop=1, this rescales
    the image so that the image fits into the target size and
    fills the boundary in according to `mode`."""
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


def intlist_to_hotonelist(cs,nc):
    """Helper function for LSTM-based OCR: encode ground truth as array.

    Given a list of target classes `cs` and a total
    maximum number of classes, compute an array that has
    a `1` in each column and time step corresponding to the
    target class."""
    result = np.zeros((2*len(cs)+1,nc))
    for i,j in enumerate(cs):
        result[2*i,0] = 1.0
        result[2*i+1,j] = 1.0
    result[-1,0] = 1.0
    return result

def hotonelist_to_intlist(outputs,threshold=0.7,pos=0):
    """Helper function for LSTM-based OCR: decode LSTM outputs.

    Translate back. Thresholds on class 0, then assigns the maximum class to
    each region. ``pos`` determines the depth of character information returned:
        * `pos=0`: Return list of recognized characters
        * `pos=1`: Return list of position-character tuples
        * `pos=2`: Return list of character-probability tuples
     """
    labels,n = measurements.label(outputs[:,0]<threshold)
    mask = np.tile(labels.reshape(-1,1),(1,outputs.shape[1]))
    maxima = measurements.maximum_position(outputs,mask,np.arange(1,np.amax(mask)+1))
    if pos==1: return maxima # include character position
    if pos==2: return [(c, outputs[r,c]) for (r,c) in maxima] # include character probabilities
    return [c for (r,c) in maxima] # only recognized characters


def spliturl(url):
    """Split a URL into its extension and base."""
    match = re.search(r"^(.+)\.([^:/]+)$", url)
    if match:
        return match.group(1), match.group(2)
    else:
        return url, ""


url_rewriter = None


def load_url_rewriter(path=None):
    """Loads a Python module that rewrites URLs names.

    This loads the rewriter from DLP_REWRITER or from $HOME/.dlp_rewriter
    """
    global url_rewriter
    if url_rewriter is not None:
        return
    if path is None:
        path = os.getenv("DLP_REWRITER", path)
    if path is None:
        path = os.path.join(os.environ.get("HOME", "/"), ".dlp_rewriter")
    if not os.path.exists(path):
        def url_rewriter(x): return x
        return
    mod = imp.load_source("url_rewriter", path)
    assert "rewriter" in dir(
        mod), "no `rewriter` function found in " + pathname
    assert isinstance(
        mod.rewriter, types.FunctionType), "rewriter is not a function"
    url_rewriter = mod.rewriterlsju


def findurl(url):
    """Finds a URL using environment variables for helpers.

    Loads DLP_REWRITER (if any) and applies it to the url. Then,
    if DLP_URLBASE is set, it reinterprets the URL relative to that
    base. Returns the new URL.
    """

    load_url_rewriter()
    orig = url
    url = url_rewriter(url)
    assert isinstance(
        url, str), "url_rewriter({}) returned {}".format(orig, url)
    base = os.environ.get("DLP_URLBASE", None)
    if base is not None:
        url = urlparse.urljoin(base, url)
    return url


def openurl(url):
    url = findurl(url)
    return urllib2.urlopen(url)


def read_shards(url, shardtype="application/x-tgz", urlpath=None, verbose=True):
    """Read a shards description file from a URL."""
    urlpath = urlpath or [""]
    if isinstance(urlpath, str):
        urlpath = urlpath.strip().split()
    shards = None
    for base in urlpath:
        trial = urlparse.urljoin(base, url)
        if verbose: print "trying: {}".format(trial)
        try:
            shards = simplejson.loads(openurl(trial).read())
            url = trial
        except urllib2.URLError:
            if verbose: print "FAILED"
            continue
    if shards is None:
        raise Exception("cannot find {} on {}".format(url, urlpath))
    if shardtype is not None and "shardtype" in shards:
        assert shards["shardtype"] == "application/x-tgz", shards["shardtype"]
    shards = shards["shards"]
    for s in shards:
        for i in range(len(s)):
            s[i] = urlparse.urljoin(url, s[i])
    return shards

###
### Data sources.
###


@itsource
def itinfinite(sample):
    """Repeat the same sample over and over again (for testing)."""
    while True:
        yield sample


@itsource
def itrepeat(source, nrepeats=int(1e9)):
    """Repeat data from a source (returned by a callable function)."""
    for i in xrange(nrepeats):
        data = source()
        for sample in data:
            yield sample

def check_ds_size(ds, size):
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
    """Iterate of training samples in a directory tree."""
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
    """Iterate over training samples given as basenames and extensions."""
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
    """Iterate over training samples given by a tabular input."""
    if isinstance(size, int): size = (size, size)
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


@itsource
def ittarreader(archive):
    """Read samples from a tar archive, either locally or given by URL.

    Batches are dictionaries using extensions as keys and containing
    file contents as values.
    """
    if isinstance(archive, str):
        if re.match(r"^(https?|file):(?i)", archive):
            archive = urllib2.urlopen(archive)
    current_prefix = None
    current_sample = None
    if isinstance(archive, str):
        stream = tarfile.open(archive, mode="r:*")
    else:
        stream = tarfile.open(fileobj=archive, mode="r|*")
    for tarinfo in stream:
        if not tarinfo.isreg():
            continue
        file = tarinfo.name
        prefix, suffix = splitallext(file)
        if prefix != current_prefix:
            if current_sample is not None:
                yield current_sample
            current_prefix = prefix
            current_sample = dict(__key__=prefix)
        data = stream.extractfile(tarinfo).read()
        current_sample[suffix] = data
    if len(current_sample.keys()) > 0:
        yield current_sample


@itsource
def itshardnames(url, shardtype="application/x-tgz",
                 randomize=True, epochs=1, urlpath=None):
    """An iterator over shard names."""
    epochs = int(epochs)
    shards = read_shards(url, shardtype=shardtype, urlpath=urlpath)
    for i in xrange(epochs):
        l = list(shards)
        if randomize:
            pyr.shuffle(l)
        for s in l:
            yield pyr.choice(s)


@itsource
def ittarshards(url, shardtype="application/x-tgz", randomize=True, epochs=1,
                urlpath=None):
    """Read a sharded data set, using a JSON-format shards file to find the shards."""
    epochs = int(epochs)
    shards = read_shards(url, shardtype=shardtype, urlpath=urlpath)
    for i in xrange(epochs):
        l = list(shards)
        if randomize:
            pyr.shuffle(l)
        for s in l:
            u = pyr.choice(s)
            for item in ittarreader(u):
                item["__epoch__"] = epoch
                yield item


@itsource
def itsqlite(dbfile, table="train", epochs=1, cols="*", extra="", verbose=False):
    """Read a dataset from an sqlite3 dbfile and the given table.

    Returns samples as dictionaries, with column names as keys
    and column contents as values (values as returned by sqlite3).
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
    """Read a dataset from an OCRopus-style book directory."""
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


###
### Basic Filters
###


@itfilter
def itinfo(data, every=0):
    """Print info about samples.

    By default only prints the first sample, but with
    `every>0`, prints `every` samples.
    """
    count = 0
    for sample in data:
        if (count == 0 and every == 0) or (every > 0 and count % every == 0):
            print "# itinfo", count
            for k, v in sample.items():
                print k,
                if isinstance(v, np.ndarray):
                    print v.dtype, v.shape
                elif isinstance(v, (str, unicode)):
                    print v[:20]
                elif isinstance(v, (int, float)):
                    print v
                elif isinstance(v, buffer):
                    print type(v), len(v)
                else:
                    print type(v), repr(v)[:20]
        count += 1
        yield sample


@itfilter
def itgrep(source, **kw):
    """Select samples from the source that match given patterns.

    Arguments are of the form `fieldname="regex"`.
    If `_not=True`, samples matching the pattern are rejected.
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
def itren(data, **kw):
    """Rename and select fields using new_name="old_name" keyword arguments."""
    for sample in data:
        sample = {k: sample[v] for k, v in kw.items()}
        yield sample


@itfilter
def itmap(data, **keys):
    """Map the fields in each sample using name=function arguments.

    Unmentioned fields are left alone.
    """
    for sample in data:
        sample = sample.copy()
        for k, f in keys.items():
            sample[k] = f(sample[k])
        yield sample


@itfilter
def itshuffle(data, bufsize=1000):
    """Shuffle the data in the stream.

    This uses a buffer of size `bufsize`. Shuffling at
    startup is less random; this is traded off against
    yielding samples quickly."""
    buf = []
    for sample in data:
        if len(buf) < bufsize:
            buf.append(data.next())
        k = pyr.randint(0, len(buf) - 1)
        sample, buf[k] = buf[k], sample
        yield sample


@itfilter
def itbatch(data, bs=20, tensors=True, partial=True):
    """Create batches of the given size."""
    batch = []
    for sample in data:
        if len(batch) >= bs:
            yield samples_to_batch(batch, tensors=tensors)
            batch = []
        batch.append(sample)
    if len(batch) == 0:
        return
    elif len(batch) == bs or partial:
        yield samples_to_batch(batch, tensors=tensors)


def maybe_index(v, i):
    try:
        return v[i]
    except:
        return v

@itfilter
def itunbatch(data):
    """Create batches of the given size."""
    for sample in data:
        keys = sample.keys()
        bs = len(sample[keys[0]])
        for i in xrange(bs):
            yield {k: maybe_index(v, i) for k, v in sample.items()}

@itfilter
def itslice(*args):
    """A pipable version of itertools.islice."""
    return itertools.islice(*args)

@itmapper
def itstandardize(sample, size, keys=["image"], crop=0, mode="nearest", augment=None):
    """Standardize images in a sample."""
    if augment:
        affine = random_affine(ralpha=augment.get(
            "ralpha"), rscale=augment.get("rscale"))
    else:
        affine = np.eye(2)
    for key in keys:
        sample[key] = standardize(
            sample[key], size, crop=crop, mode=mode, affine=affine)
    if augment:
        for key in keys:
            sample[key] = random_gamma(sample[key],
                                       rgamma=augment.get("rgamma"),
                                       cgamma=augment.get("cgamma"))
    return sample

###
### Specialized input pipelines for OCR, speech, and related tasks.
###

def makeseq(image):
    """Turn an image into an LD sequence."""
    assert isinstance(image, np.ndarray), type(image)
    if image.ndim==3 and image.shape[2]==3:
        image = np.mean(image, 2)
    assert image.ndim==2
    return image.T

def makebatch(images, for_target=False):
    """Given a list of LD sequences, make a BLD batch tensor."""
    assert isinstance(images, list), type(images)
    assert isinstance(images[0], np.ndarray), images
    assert images[0].ndim==2, images[0].ndim
    l, d = np.amax(np.array([img.shape for img in images], 'i'), axis=0)
    ibatch = np.zeros([len(images), int(l), int(d)])
    if for_target:
        ibatch[:, :, 0] = 1.0
    for i, image in enumerate(images):
        l, d = image.shape
        ibatch[i, :l, :d] = image
    return ibatch

def images2batch(images):
    """Given a list of images, return a BLD batch tensor."""
    images = [makeseq(x) for x in images]
    return makebatch(images)

class AsciiCodec(object):
    """An example of a codec, used for turning strings into tensors."""
    def _encode_char(self, c):
        if c=="": return 0
        return max(1, ord(c) - ord(" ") + 1)
    def _decode_char(self, c):
        if c==0: return ""
        return chr(ord(" ") + c - 1)
    def size(self):
        """The number of classes. Zero is always reserved for the empty class."""
        return 97
    def encode(self, s):
        """Encode a string."""
        return [self._encode_char(c) for c in s]
    def decode(self, l):
        """Decode a numerical encoding of a string."""
        return "".join([self._decode_char(x) for x in l])

ascii_codec = AsciiCodec()

def maketarget(s, codec=ascii_codec):
    """Turn a string into an LD target."""
    assert isinstance(s, (str, unicode)), (type(s), s)
    codes = codec.encode(s)
    n = codec.size()
    return intlist_to_hotonelist(codes, n)

def transcripts2batch(transcripts, codec=ascii_codec):
    """Given a list of strings, makes ndarray target arrays."""
    targets = [maketarget(s, codec=codec) for s in transcripts]
    return makebatch(targets, for_target=True)

@itfilter
def itbatchedbuckets(data, batchsize=5, scale=1.8, seqkey="input", batchdim=1):
    """List-batch input samples into similar sized batches."""
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
def itlinebatcher(data, input="input", target="target", codec=ascii_codec):
    """Performs text line batching for OCR."""
    for sample in data:
        sample = sample.copy()
        sample[input] = images2batch(sample[input])
        sample[target] = transcripts2batch(sample[target], codec=codec)
        yield sample

###
### various data sinks
###

@itsink
def showgrid(data, key="input", label=None, grid=(4, 8)):
    """A sink that shows a grid of images."""
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
def showgrid2(data, key="input", value="target", label=None, grid=(4, 8)):
    """A sink that shows a grid of pairs of images."""
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
    """A sink that provides info about samples/batches."""
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
