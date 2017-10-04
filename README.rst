Machine Learning Pipelines
==========================

Deep learning usually contains at its core logic like the following:

::

        net = make_network(...)
        for epoch in xrange(num_epochs):
            for sample in read_training_samples("./dataset"):
                sgd_train(net, sample["input"}, sample["target"])
            cost = 0
            for sample in read_test_samples("./testset"):
                cost += sgd_eval(net, sample["input"], sample["target"])
            print epoch, cost

The ``dlpipes`` library is intended to make writing the
``read_training_samples`` and ``read_test_samples`` functions easy and
efficient.

In fact, what is expressed as ``read_training_samples`` above usually
contains a number of processing steps:

-  open the file system or some kind of record file
-  shuffle the order of training samples
-  decompress any compressed images
-  normalize image sizes
-  perform data augmentation by random transformations
-  batch up the data into batches of a desired size

Often, these kinds of input pipelines are written as length pieces of
procedural code, but that makes code reuse difficult and makes it
difficult to understand and modify such pipelines. Using ``dlinputs``,
such a complex input pipeline can simply be written as:

::

        with dlinputs.ops:
            data = itfiles(root_dir, "png,cls") | \
                   itshuffle(1000) | \
                   itmap(png=pnggray, cls=int) | \
                   itren(image="png", cls="cls") | \
                   itstandardize((224, 224), "png") | \
                   itdistort([5, 5]) | \
                   itbatch(20)

        for sample in data:
            sgd_train(net, sample["input"}, sample["target"])

In this code, the statement ``with dlinputs.ops`` simple *temporarily*
imports the input operators from the ``dlinputs`` package; we could just
as well have said ``dlinputs.itfiles`` etc.

All the elements of this pipeline (``itfiles``, ``itshuffle``, etc.) are
just simple python functions that internally look something like:

::

        def pipeline_stage(data, parameters, ...):
            for sample in data:
                yield transformed(sample)

Pipeline stages can also be executed in parallel, or even distributed
across multiple nodes or containers; we will talk about that later.

Sharded Tar Files
=================

Format
------

Large machine learning datasets are usually broken up into pieces
of size 10M - 10G called _shards_. Data within each shard is
usually processed sequentially.

- sequential reads/writes are much more efficient than random access
- by shuffling shards, we can randomize DL inputs and still enjoy sequential access
- shards can be processed in parallel for map-reduce jobs

The `dlinputs` library uses tar files as its main dataset storage format; this
has the advantage that data is represented in the same way as it is on disk
and that data can be manipulated directly using standard tools.
However, other formats are supported as well, including directory trees,
file lists, SQLite databases, and anything that yields a Python iterator.
(Direct support for video I/O, `tf.Record`/`tf.Example`, and MsgPack is
also planned.)

For example, to turn an on-disk dataset into a tar files suitable for
training, just use:

::
        find . -iname '*.png' -o -iname '*.cls' | sort |
            tar -ztvf data.tgz -T -

With sharding, use the included `tarshards` program:

::
        find . -iname '*.png' -o -iname '*.cls' | sort |
            tarshards data

This will now create shards with names like `data-000000.tgz`.

To iterate over this data, you can now use the input pipeline:

::

        with dlinputs.ops:
            data = ittarfile("data.tgz") | \
                   itshuffle(1000) | \
                   ... same pipeline as above ...

Since this is just sequential data, you can also stream this data from a
web server:

::

        with dlinputs.ops:
            data = ittarfile("http://eunomia/data.tgz") | \
                   itshuffle(1000) | \
                   ... same pipeline as above ...

To iterate over sharded data, use a url of the form `data-@000123.tgz`,
where the number of shards is given after the `@` sign:

::

        with dlinputs.ops:
            data = ittarshards("http://eunomia/data-@000123.tgz") | \
                   itshuffle(1000) | \
                   ... same pipeline as above ...

The ``ittarshards`` iterator can randomization and load balancing; it
performs roughly the following operations:

-  shuffle the list of shards
-  for each shard, randomly pick a URL from the list of URLs
-  iterate through the tar file given by the URL like ``ittarfile``

Note that a high performance web server for sharded tar files will
redirect the URLs for each shard to different servers.

Shard Writing
-------------

In addition to training DL models from sharded tar files, another very
common operation is dataset transformations. Such transformations are
supported by the ``ShardWriter`` class.

::

        writer = shardwriter.ShardWriter("result",
                                          converters=...,
                                          names=...,
                                          shardsize=1e8)
        for batch in source:
            writer.write(batch["key"], batch)

(For parallelizing such transformations for large datasets, there will
eventually be additional tools.)

Common Pipeline Operations
==========================

Data Sources
------------

The ``dlinputs`` library provides a number of common input sources:

-  ``itfiles`` -- files and directories
-  ``itsqlite`` -- SQLite data sources
-  ``ittarfile`` -- tar files (including from URLs)
-  ``ittarshards`` -- sharded tar files (including from URLs)

Data Transformations
--------------------

-  ``itshuffle`` -- shuffle samples
-  ``itren`` -- select and rename input fields
-  ``itmap`` -- apply functions to input fields
-  ``itbatch`` -- build batches from samples
-  ``itbatchedbuckets`` -- build batches from similarly sized samples

Data Augmentation
-----------------

-  ``itstandardize`` -- resize to a standard size, optionally augment
-  ``itdistort`` -- agument by nonlinear distortions

Pipelines as Composition of Iterators
=====================================

The code contained within the ``with dlinputs.ops:`` block behaves very
much like a UNIX pipeline. It constists of two kinds of components:

-  ``itfiles`` is a data *source*
-  ``itshuffle``, ``itmap``, ... are *filters*

Note that the result of any of these pipeline operations is simply a
Python *iterator*. By convention, the objects that we iterate over are
dictionaries with string keys and values that are usually strings,
tensors, or numbers. That is, the ``itfiles`` function call above
corresponds roughly to a function like this:

::

        def itfiles(...):
            for fname, fname2 in find_filenames(...):
                yield dict(png=open(fname).read(),
                           cls=open(fname2).read())

The ``itmap`` call corresponds roughly to the following function:

::

        def itmap(...):
            def mapper(data):
                for sample in data:
                    yield dict(png=pnggray(sample["png"]),
                               cls=int(sample["cls"]))
            return mapper

In fact, if you want to write your own filter, ``dlinputs`` provides a
simple notation that allows you to do so without the currying. For
example, here is a simple filter that selects all records containing the
given fields:

::

        @dlinputs.itfilter
        def select_image(data, fields):
            for sample in data:
                if all(field in sample for field in fields):
                    yield sample

You can now write the following (note that the ``@dlinputs.itfilter``
decorator has implicitly curried the function so that the first
argument, ``data`` is not explicit anymore):

::

        data = itfiles(root_dir, "png,cls") | \
               itfilter(["png", "cls"]) | \
               ...

Planned Additions
=================

We're planning the following additional features:

- iterate over `tf.Record`/`tf.Example` files
- iterate over concatenated MsgPack data
