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
``read_training_samples`` and ``read_test_samples`` functions simple
while enabling the use of petascale datasets based on
simple web technologies.

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

        source = gopen.sharditerator("http://server/data-@000123.tgz")
        pipeline = filters.compose(
            filters.shuffle(1000),
            filters.rename(input="png", target=".out.png"),
            filters.batched(10))
        for sample in pipeline(source):
            sgd_train(net, sample["input"], sample["target"])

All the elements of this pipeline (``shuffle``, ``rename``, ``batched``, etc.) are
just simple python functions that internally look something like:

::
        @curried
        def pipeline_stage(data, parameters, ...):
            for sample in data:
                yield transformed(sample)

or equivalently

::
        def pipeline_stage(parameters, ...):
            def iterator(data):
                for sample in data:
                    yield transformed(sample)
            return iterator


Sharded Tar Files
=================

Format
------

Large machine learning datasets are usually broken up into pieces
of size 10M - 10G called *shards*. Data within each shard is
usually processed sequentially. Using shards has several advantages:

- sequential reads/writes are much more efficient than random access
- by shuffling shards, we can randomize DL inputs and still enjoy sequential access
- shards can be processed in parallel for map-reduce jobs

The ``dlinputs`` library uses tar files as its main dataset storage format; this
has the advantage that data is represented in the same way as it is on disk
and that data can be manipulated directly using standard tools.
However, other formats are supported as well, including directory trees,
file lists, SQLite databases, and anything that yields a Python iterator.
(Direct support for video I/O, ``tf.Record``/``tf.Example``, and MsgPack is
also planned.)

For example, to turn an on-disk dataset into a tar files suitable for
training, just use:

::

        tar -ztvf data.tgz --sort data

For more complex selection of files, or if your ``tar`` doesn't support ``--sort``,
you can also write:

::

        find data -iname '*.png' -o -iname '*.cls' | sort | \
            tar -ztvf data.tgz -T -

With sharding, use the included ``tarshards`` program:

::

        find . -iname '*.png' -o -iname '*.cls' | sort | \
            tarshards -s 1e7 data

This will now create shards with names like ``data-000000.tgz`` and a
shard size of about 10 MB.  (Picking shard sizes involves tradeoffs
between I/O efficiency, parallelism, and randomization of datasets,
but it's a good idea to pick shard sizes that are at least 10 MB big
and aim for at least a few dozen shards. Small datasets can otherwise
just be stored unsharded.)

To iterate over this data, you can now use the input pipeline:

::

        for sample in gopen.sharditerator("data-@000010.tgz"):
            ...

The ``sharditerator`` can perform randomization and load balancing by
performing roughly the following operations:

-  shuffle the list of shards
-  for each shard, randomly pick a URL from the list of URLs
-  iterate through the tar file given by the URL, in the same way as ``ittarfile``

Note that a high performance web server for sharded tar files will
redirect the URLs for each shard to different servers.

In addition to training DL models from sharded tar files, another very
common operation is dataset transformations. Such transformations are
supported by the ``ShardWriter`` class.

::

        writer = tarrecords.ShardWriter("result")
        for sample in source:
            sample = transform(sample)
            writer.write(sample)

Multiple Data Sources, Patching
===============================

Data for training is often composed of multiple datasets and corrections.
It's easy to express such compositions of training datasets with ``dlinputs``:

::
        ukdata = gopen.sharditerator("http://server/uk-data-@000100.tgz")
        ukdata_patched = filters.patched("http://server/uk-patches-2017-08.tgz")(ukdata)
        usdata = gopen.sharditerator("http://server/us-data-@000100.tgz")
        usdata_patched = filters.patched("http://server/us-patches-2017-09.tgz")(usdata)
        training_data = filters.merge(ukdata_patched, usdata_patched)
        batched_data = filters.batched(10)(training_data)

        for sample in batched_data:
            ...


Distributed Processing with ZMQ
=================

The ``dlinputs`` library also supports large scale distributed preprocessing
pipelines using the primitives in the ``zcom`` library. This uses MessagePack and
ZMQ for moving data between compute nodes. This code works particularly well
in environments like Kubernetes.

Command Line Tools
==================

There are a few simple command line tools:

- ``run-tests``: run tests
- ``show-input``: iterate over sharded tar files or network inputs and print/display
- ``tarshards``: create tar shards from a list of files
- ``transform-input``: simple utility for transforming sharded tar files
- ``training-test-split``: split data into training/test samples

Planned Additions
=================

We're planning the following additional features:

- iterate over ``tf.Record``/``tf.Example`` files
