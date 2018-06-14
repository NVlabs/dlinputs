#!/usr/bin/python

import argparse

import zcom
import gopen
import paths


def loader(input, output, report=0, epochs=1000000000):
    if report>0:
        print "loader", input, "->", output
    while True:
        outputs = zcom.Connection(output, encode=False)
        shards = paths.path_shards(input)
        count = 0
        for shard in shards:
            inputs = gopen.sharditerator(shard, epochs=epochs, decode=False)
            for sample in inputs:
                outputs.send(sample)
                if report>0 and count%report==0:
                    print "{:6d} {:6.1f} samples/s {:8.1f} MB/s".format(
                        count, outputs.stats.recent_rate(), outputs.stats.recent_throughput() / 1e6)
                count += 1

if __name__ == "__main__":
    parser = argparse.ArgumentParser("Load and broadcast shards.")
    parser.add_argument("-o", "--output", default="zpush://localhost:10000")
    parser.add_argument("-r", "--report", type=int, default=1000)
    parser.add_argument("input")
    args = parser.parse_args()
    loader(args.input, args.output, report=args.report)
