# Machine Learning Pipelines

Deep learning usually contains at its core logic like the following:
```
    net = make_network(...)
    for epoch in xrange(num_epochs):
        for sample in read_training_samples("./dataset"):
            sgd_train(net, sample["input"}, sample["target"])
        cost = 0
        for sample in read_test_samples("./testset"):
            cost += sgd_eval(net, sample["input"], sample["target"])
        print epoch, cost
```
The `dlpipes` library is intended to make writing the `read_training_samples` and `read_test_samples` functions easy and efficient.

In fact, what is expressed as `read_training_samples` above usually contains a number of processing steps:

- open the file system or some kind of record file
- shuffle the order of training samples
- decompress any compressed images
- normalize image sizes
- perform data augmentation by random transformations
- batch up the data into batches of a desired size

Often, these kinds of input pipelines are written as length pieces of procedural code, but that makes code reuse difficult and makes it difficult to understand and modify such pipelines. Using `dlinputs`, such a complex input pipeline can simply be written as:

```
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
```
In this code, the statement `with dlinputs.ops` simple _temporarily_ imports the input operators from the `dlinputs` package; we could just as well have said `dlinputs.itfiles` etc.

All the elements of this pipeline (`itfiles`, `itshuffle`, etc.) are just simple python functions that internally look something like:

```
    def pipeline_stage(data, parameters, ...):
        for sample in data:
            yield transformed(sample)
```

Pipeline stages can also be executed in parallel, or even distributed across multiple nodes or containers; we will talk about that later.

# Sharded Tar Files

## The Need for Sequential Reads

Datasets larger than memory need to be read from disk, and since DL models are usually trained by epoch, caching generally helps little in terms of speeding this up.

DL jobs can easily consume data at a rate of more than 1 Gbyte / second / GPU and in order to utilize GPU hardware efficiently, we need to supply data as close to that rate as possible. Common top read speeds for modern rotational disks are about 200 MB/s [^1], and for modern flash drives are about 500 MB/s [^2]; NVMe drives clock in at up to 1.7 GB/s [^3]

[^1]: http://hdd.userbenchmark.com/WD-Black-6TB-2015/Rating/3519

[^2]: http://ssd.userbenchmark.com/HyperX-Savage-480GB/Rating/3602

[^3]: http://www.storagereview.com/samsung_960_pro_m2_nvme_ssd_review

However, training samples for DL problems individually tend to be fairly small (a few kbytes to a few hundred kbytes). If we store these training samples in individual files on a file system, each sample requires multiple seeks and performance drops dramatically, both for rotational disks (4k random reads: 200 MB/s $\rightarrow$ 4 MB/s) and SSD (500 MB/s $\rightarrow$ 24 MB/s, 1.7 GB/s $\rightarrow$ 52 MB/s).

The fact that sequential reads are much faster than random access reads is a long-standing phenomenon across many storage architectures (starting with tape drives). It is therefore well understood how to deal with such data:

- data is stored in sequential record format
- data is striped or sharded across many different storage devices
- data is generally consumed sequentially

The TensorFlow system uses `tf.Record` for its sequential record storage, and encodes each record as a `tf.Example` protocol buffer. These are Google-proprietary formats that have few tools available for them outside of Google.

## Tar Files as Record Files

The `dlinputs` library adopts as its record format the well-known UNIX `tgz` format, gzip-compressed `tar` files. There is a large number of command line tools and libraries available for reading and writing such files. For example, if you have MNIST or CIFAR training data in a directory, you can create a `tgz` file suitable for sequential training with the UNIX command:

```
    find . -iname '*.png' -o -iname '*.cls' | sort | 
        tar -ztvf data.tgz -T -
```

To iterate over this data, you can now use the input pipeline:

```
    with dlinputs.ops:
        data = ittarfile("data.tgz") | \
               itshuffle(1000) | \
               ... same pipeline as above ...
```

Since this is just sequential data, you can also stream this data from a web server. If you copy it to `/var/www/html` on host `eunomia`, you can then write:

```
    with dlinputs.ops:
        data = ittarfile("http://eunomia/data.tgz") | \
               itshuffle(1000) | \
               ... same pipeline as above ...
```

Note that this pipeline does _not_ download and unpack the training data; rather, it streams the data from the web server directly and starts training as soon as the first complete training sample has been received.

The same data also works from an S3 storage server:

```
    with dlinputs.ops:
        data = ittarfile("http://s3-aws-region.amazonaws.com/bucket/data.tgz") | \
               itshuffle(1000) | \
               ... same pipeline as above ...
```

## Sharding

The sequential read bandwidth of individual low-cost drives is about 200 MB/s, far below what is needed to keep GPUs busy. To keep large DL training jobs busy, it is therefore important to perform input from multiple drives simultaneously. Traditionally, RAID systems or sophisticated file system types have been used, but the most common solution used for DL is much simpler: sharding. Sharding not only allows parallel I/O across multiple drives, it also permits easy shuffling and simple, robust parallel data transformations.

A sharded tar file is simply a tar file that has been split into approximately equal sizes at record boundaries, i.e., split such that files with the same basename but different extensions are always contained within a single shard. Common shard sizes are in the range of 100 MB to 1 GB.

Shards are described by a simple JSON file with the following format, e.g. `imagenet_train.shards`:

```
    {
        "metadata": {
        },
        "shards": [
            [
                "imagenet_train-000000.tgz"
            ],
            [
                "imagenet_train-000001.tgz"
            ],
            [
                "imagenet_train-000002.tgz"
            ],
            ...
        ]
    }
```

Shards are described by a list of sublists; each sublist contains URLs where that shard can be found. Most commonly, this is just one element that is relative to the shards file. However, sublists can contain multiple URLs and those URLs can be absolute or relative.

There is a simple iterator for reading from sharded tar files as well:

```
    with dlinputs.ops:
        data = ittarshards("http://eunomia/imagenet_train.tgz") | \
               itshuffle(1000) | \
               ... same pipeline as above ...
```

The `ittarshards` iterator can randomization and load balancing; it performs roughly the following operations:

- shuffle the list of shards
- for each shard, randomly pick a URL from the list of URLs
- iterate through the tar file given by the URL like `ittarfile`

(In the future, `ittarshards` will perform parallel I/O from multiple shards at once.)

## Shard Writing

In addition to training DL models from sharded tar files, another very common operation is dataset transformations. Such transformations are supported by the `ShardWriter` class.

```
    writer = shardwriter.ShardWriter("result",     
                                      converters=...,          
                                      names=...,          
                                      shardsize=1e8)
    for batch in source:
        writer.write(batch["key"], batch)
```

(For parallelizing such transformations for large datasets, there will eventually be additional tools.)

# Common Pipeline Operations

## Data Sources

The `dlinputs` library provides a number of common input sources:

- `itfiles` -- files and directories
- `itsqlite` -- SQLite data sources
- `ittarfile` -- tar files (including from URLs)
- `ittarshards` -- sharded tar files (including from URLs)

## Data Transformations

- `itshuffle` -- shuffle samples
- `itren` -- select and rename input fields
- `itmap` -- apply functions to input fields
- `itbatch` -- build batches from samples
- `itbatchedbuckets` -- build batches from similarly sized samples

## Data Augmentation

- `itstandardize` -- resize to a standard size, optionally augment
- `itdistort` -- agument by nonlinear distortions

# Distributed Pipelines

The `dlinputs` library by itself provides a convenient way of accessing datasets in common formats and to manipulate the data before training. However, Python is single threaded and processing one sample at a time in the input pipeline may not be fast enough. The `dlinputs` library works fine with `multiprocessing` or `torch.multiprocessing`, so that is an easy way of running `dlinputs` pipelines on multiple cores. However, in many cases, distributed pipelines are preferable, that is, running parts of an input pipeline on multiple nodes.

The basic support for distributed input pipelines are `itzmq` for connecting network sources to pipelines on the input, and `zmqserver` for outputting samples to other clients. 

To iterate over data from a network-based server, use a pipeline like this:

```
    with dlinputs.ops:
        data = itzmq("server:7000") | itshuffle(1000) | ...   
    for sample in data:
        ...
```

To serve data to be consumed by other clients, use a pipeline like this:

```
    with dlinputs.ops:
        data = ittarfile("data.tgz")
    zmqserver(data, bind="*:7000")
```

As the names suggest, `itzmq` and `zmqserver` use the ZMQ message queue protocol for distributing data. The default encoding used allows efficient distribution and manipulation of tensor data. In particular, tensors are encoded as separate ZMQ memory buffers, allowing them to be moved and used without copying in clients. ZMQ is supported by many different languages, making it easy to write tools, servers, and clients in languages other than Python.

The use of ZMQ permits building very efficient pipelines for many large scale DL problems. For example, a common problem is training a large number of models with different hyperparameters on the same dataset. This can be handled efficiently via ZMQ PUB/SUB sockets.

Here is a DL training pipeline that combines parallel preprocessing with PUB/SUB training data distribution:

\includegraphics[scale=0.4]{dlinputs-zmq.png}

Here, the preprocessing/augmentation processes access sharded tar files on web servers. The result of the preprocessing is then sent to a PUB/SUB server (this can also be multithreaded). Multiple GPU jobs then subscribe to samples from the PUB/SUB server. The key advantage of this pipeline is that the web server and preprocessing pipeline only need to yield samples at the rate of a single DL training job; the PUB/SUB system then distributes these samples to all DL training jobs.

With Docker Compose or Kubernetes, such complex training pipelines can be specified in a single job specification.

```
    services:
        augment:
            image: tmbdev/pytorch-full
            command: imagenet-augment http://ceph/coco.shards pubsub:7000
            expose: ["7000"]
            replicas: 8
        pubsub:
            image: tmbdev/pytorch-full
            command: pubsub-server '*:7000' '*:7001'
        dljob:
            image: tmbdev/pytorch-full
            command: dltrain pubsub:7000
            replicas: 16
```

Note how all three commands (`imagenet-augment`, `pubsub-server`, and `dltrain`) refer to each other by job name within the Compose file, and how the number of replicas is specified directly in the Docker Compose file.

# Other Formats

Although sharded tar files work well for many kinds of large scale learning, some datatypes, such as medical images and videos, are large enough not to require sharding.

Generally, such input pipelines are described by JSON files similar to sharded record files, with some additional conventions and the ability to incorporate class data directly into the JSON file:

```
    {
        "metadata": {
        },
        "videos": [
            {
                "video_URL": "video1.mp4",
                "segmentation_URL", "segmetation1.mp4",
                "cls": 17
            },
            {
                "video_URL": "video2.avi",
                "segmentation_URL", "segmentation2.avi",
                "cls": 193
            },
            ...
        ]
    }
```

Video I/O code hasn't been merged into the first version of `dlinputs` but will be merged in upcoming releases.

# Pipelines as Composition of Iterators

The code contained within the `with dlinputs.ops:` block behaves very much like a UNIX pipeline. It constists of two kinds of components:

- `itfiles` is a data _source_
- `itshuffle`, `itmap`, ... are _filters_

Note that the result of any of these pipeline operations is simply a Python _iterator_. By convention, the objects that we iterate over are dictionaries with string keys and values that are usually strings, tensors, or numbers. That is, the `itfiles` function call above corresponds roughly to a function like this:

```
    def itfiles(...):
        for fname, fname2 in find_filenames(...):
            yield dict(png=open(fname).read(),
                       cls=open(fname2).read())
```

The `itmap` call corresponds roughly to the following function:

```
    def itmap(...):
        def mapper(data):
            for sample in data:
                yield dict(png=pnggray(sample["png"]),
                           cls=int(sample["cls"]))
        return mapper
```

In fact, if you want to write your own filter, `dlinputs` provides a simple notation that allows you to do so without the currying. For example, here is a simple filter that selects all records containing the given fields:

```
    @dlinputs.itfilter
    def select_image(data, fields):
        for sample in data:
            if all(field in sample for field in fields):
                yield sample
```

You can now write the following (note that the `@dlinputs.itfilter` decorator has implicitly curried the function so that the first argument, `data` is not explicit anymore):

```
    data = itfiles(root_dir, "png,cls") | \
           itfilter(["png", "cls"]) | \
           ...
```
