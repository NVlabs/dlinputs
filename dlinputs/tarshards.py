
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


def tarreader(archive, check_sorted=True, keys=base_plus_ext, decoder=lambda x: x):
    """Iterate over samples from a tar archive, either locally or given by URL.

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
                yield decoder(current_sample)
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
        yield decoder(current_sample)
    try: del stream
    except: pass
    try: del archive
    except: pass


@itsource
def ittarreader(archive, epochs=1, **kw):
    for epoch in xrange(epochs):
        source = ittarreader1(archive, **kw)
        count = 0
        while True:
            try:
                sample = source.next()
                sample["__epoch__"] = epoch
                count += 1
                yield sample
            except StopIteration:
                break
            except Exception, e:
                print "ittarreader archive:", archive
                print "ittarreader Exception:", e
                print "quitting after", count, "records"
                break

def tarreader(archive, epochs=1, **kw):
    for epoch in xrange(epochs):
        source = ittarreader1(archive, **kw)
        count = 0
        while True:
            try:
                sample = source.next()
                sample["__epoch__"] = epoch
                count += 1
                yield sample
            except StopIteration:
                break
            except Exception, e:
                print "ittarreader archive:", archive
                print "ittarreader Exception:", e
                print "quitting after", count, "records"
                break

@itsource
def tarshards(url, shardtype="application/x-tgz", randomize=True, epochs=1,
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

