# Copyright (c) 2017 NVIDIA CORPORATION. All rights reserved.
# See the LICENSE file for licensing terms (BSD-style).

import os
import os.path
import StringIO
import tarfile
import warnings
import re
import time
import getpass
import socket

import utils

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

def trivial_decode(sample):
    result = {}
    for k, v in sample.items():
        if isinstance(v, buffer):
            v = str(v)
        elif isinstance(v, unicode):
            v = str(codecs.encode(v, "utf-8"))
        else:
            assert isinstance(v, str)
        result[k] = v
    return result

def valid_sample(sample):
    return (sample is not None and
            len(sample.keys()) > 0 and
            not sample.get("__bad__", False))

def group_by_keys(keys=base_plus_ext, lcase=True):
    """Groups key, value pairs into samples."""
    def iterator(data):
        current_count = 0
        current_sample = None
        for fname, value in data:
            prefix, suffix = keys(fname)
            if prefix is None:
                continue
            if current_sample is not None and prefix == current_sample["__key__"]:
                current_sample[suffix] = value
                continue
            if valid_sample(current_sample):
                yield current_sample
            current_sample = dict(__key__=prefix)
            if lcase:
                suffix = suffix.lower()
            current_sample[suffix] = value
        if valid_sample(current_sample):
            yield current_sample
    return iterator

def tardata(fileobj):
    """Iterator yielding filename, content pairs for the given tar stream."""
    stream = tarfile.open(fileobj=fileobj, mode="r|*")
    for tarinfo in stream:
        if not tarinfo.isreg(): continue
        fname = tarinfo.name
        if fname is None: continue
        data = stream.extractfile(tarinfo).read()
        yield fname, data
    del stream

def decoder(decode=True):
    """Apply tariterator-like decoding to the stream of samples."""
    if decode is True:
        decode = utils.autodecode
    elif decode is False:
        decode = trivial_decode
    def iterator(data):
        for sample in data:
            yield decode(sample)
    return iterator

def tariterator1(fileobj, check_sorted=False, keys=base_plus_ext, decode=True):
    """Alternative (new) implementation of tariterator."""
    content = tardata(fileobj)
    samples = group_by_keys(keys=keys)(content)
    decoded = decoder(decode=decode)(samples)
    return decoded

def zipdata(fname):
    """Iterator yielding filename, content pairs for the given zip file."""
    import zipfile
    zf = zipfile.ZipFile(fname)
    fnames = sorted(zf.namelist())
    for fname in fnames:
        data = zf.open(fname).read()
        yield fname, data

def zipiterator(fname, check_sorted=False, keys=base_plus_ext, decode=True):
    content = zipdata(fname)
    samples = group_by_keys(keys=keys)(content)
    decoded = decoder(decode=decode)(samples)
    return decoded

def tariterator(fileobj, check_sorted=False, keys=base_plus_ext, decode=True, source=None, lcase=True):
    """Iterate over samples from a tar archive, either locally or given by URL.

    Tar archives are assumed to be sorted by file name. For each basename,
    reads all the files with different extensions and returns a dictionary
    with the extension as key and the file contents as value.

    :param str archive: tar archive with sorted file names (file name or URL)
    :param bool check_sorted: verify that file names are sorted
    :returns: iterator over samples

    """
    if decode is True:
        decode = utils.autodecode
    elif decode is False:
        decode = trivial_decode
    current_count = 0
    current_prefix = None
    current_sample = None
    stream = tarfile.open(fileobj=fileobj, mode="r|*")
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
            if valid_sample(current_sample):
                yield decode(current_sample)
            current_prefix = prefix
            current_sample = dict(__key__=prefix, __source__=source)
        try:
            data = stream.extractfile(tarinfo).read()
        except tarfile.ReadError, e:
            print "tarfile.ReadError at", current_count
            print "file:", tarinfo.name
            print e
            current_sample["__bad__"] = True
        else:
            if lcase:
                suffix = suffix.lower()
            current_sample[suffix] = data
            current_count += 1
    if valid_sample(current_sample):
        yield decode(current_sample)
    try: del stream
    except: pass

class TarWriter(object):
    def __init__(self, fileobj, keep_meta=False, encode=True, user=None, group=None):
        """A class for writing dictionaries to tar files.

        :param fileobj fileobj: file name for tar file (.tgz)
        :param bool keep_meta: keep fields starting with "_"
        :param function encoder: encoding of samples prior to writing
        """
        if isinstance(fileobj, str):
            fileobj = open(fileobj, "wb")
        if encode is True:
            encode = utils.autoencode
        elif encode is False:
            encode = lambda x: x
        self.keep_meta = keep_meta
        self.encode = encode
        self.stream = fileobj
        self.tarstream = tarfile.open(fileobj=fileobj, mode="w:gz")
        self.user = user or getpass.getuser()
        self.group = group or socket.gethostname()

    def __enter__(self):
        """Context manager."""
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager."""
        self.finish()

    def close(self):
        """Close the tar file."""
        self.finish()

    def finish(self):
        """Close the tar file."""
        self.tarstream.close()
        if self.stream:
            self.stream.close()
            self.stream = None

    def write(self, obj):
        """Write a dictionary to the tar file.

        :param str obj: dictionary of objects to be stored
        :returns: size of the entry

        """
        total = 0
        obj = self.encode(obj)
        assert "__key__" in obj, "object must contain a __key__"
        for k, v in obj.items():
            if k[0]=="_": continue
            assert isinstance(v, str), "{} doesn't map to a string after encoding ({})".format(k, type(v))
        key = obj["__key__"]
        for k in sorted(obj.keys()):
            if not self.keep_meta and k[0]=="_":
                continue
            v = obj[k]
            assert isinstance(v, (str, buffer)),  \
                "converter didn't yield a string: %s" % ((k, type(v)),)
            now = time.time()
            ti = tarfile.TarInfo(key + "." + k)
            ti.size = len(v)
            ti.mtime = now
            ti.mode = 0o666
            ti.uname = "bigdata"
            ti.gname = "bigdata"
            stream = StringIO.StringIO(v)
            self.tarstream.addfile(ti, stream)
            total += ti.size
        return total

class ShardWriter(object):
    def __init__(self, pattern, maxcount=100000, maxsize=3e9, keep_meta=False, encode=True, user=None, group=None):
        self.verbose = 1
        self.args = dict(keep_meta=keep_meta, encode=encode, user=user, group=group)
        self.maxcount = maxcount
        self.maxsize = maxsize
        self.tarstream = None
        self.shard = 0
        self.pattern = pattern
        self.total = 0
        self.count = 0
        self.size = 0
        self.next_stream()
    def next_stream(self):
        if self.tarstream is not None:
            self.tarstream.close()
        self.fname = self.pattern % self.shard
        if self.verbose:
            print "# writing", self.fname, self.count, "%.1f GB"%(self.size/1e9), self.total
        self.shard += 1
        stream = open(self.fname, "wb")
        self.tarstream = TarWriter(stream, **self.args)
        self.count = 0
        self.size = 0
    def write(self, obj):
        if self.tarstream is None or self.count>=self.maxcount or self.size>=self.maxsize:
            self.next_stream()
        size = self.tarstream.write(obj)
        self.count += 1
        self.total += 1
        self.size += size
    def close(self):
        self.tarstream.close()
        del self.tarstream
        del self.shard
        del self.count
        del self.size

