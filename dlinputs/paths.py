# Copyright (c) 2017 NVIDIA CORPORATION. All rights reserved.
# See the LICENSE file for licensing terms (BSD-style).

import os
import os.path
import re
import urllib2
import urlparse
from collections import namedtuple
import simplejson
from numpy import clip

def split_sharded_path(path):
    """Split a path containing shard notation into prefix, format, suffix, and number."""
    match = re.search(r"^(.*)@([0-9]+)(.*)", path)
    if not match:
        return path, None
    prefix = match.group(1)
    num = int(match.group(2))
    suffix = match.group(3)
    fmt = "%%0%dd" % len(match.group(2))
    return prefix+fmt+suffix, num

def path_shards(path):
    """Given a path shard spec, return an iterator over the shards."""
    fmt, n = split_sharded_path(path)
    if n is None:
        yield fmt
    else:
        for i in range(n):
            yield (fmt % i)

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
    for i in xrange(n):
        index = "%0*d" % (f, i)
        yield ShardEntry(prefix+index+suffix, prefix, index, suffix)

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

def writefile(fname, data):
    """Helper function to read a file (binary).

    :param fname: file name to be read
    :returns: contents of file

    """
    with open(fname, "wb") as stream:
        return stream.write(data)


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

def parse_save_path(path, extension="pt"):
    if extension is not None:
        extension = "\\." + extension
    match = re.search(r"^.*-([0-9]{6,})-([0-9]{6,})"+extension, path)
    if not match:
        return 0, None
    return int(match.group(1))*1000, int(match.group(2))*1e-6

def make_save_path(prefix, ntrain, error, extension="pt"):
    assert isinstance(ntrain, int)
    assert isinstance(error, float)
    assert ntrain < 1e12
    assert error >= -1e-6
    assert error <= 1+1e-6
    if extension is not None:
        extension = "." + extension
    error = clip(error, 0, 1)
    kilos = int(ntrain // 1000)
    micros = int(error * 1e6)
    return prefix + "-{:09d}-{:06d}".format(kilos, micros) + extension
