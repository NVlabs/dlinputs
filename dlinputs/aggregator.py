#!/usr/bin/python

import argparse
import zcom
import time

def aggregator(input, output, report=0):
    if report>0:
        print "aggregator", input, "->", output
    while True:
        try:
            inputs = zcom.Connection(input, pack=False, encode=False)
            outputs = zcom.Connection(output, pack=False, encode=False)
            count = 0
            while True:
                if report>0 and count%report==0:
                    print "{:6d} {:6.1f} samples/s {:8.1f} MB/s".format(
                        count, outputs.stats.recent_rate(), outputs.stats.recent_throughput() / 1e6)
                sample = inputs.recv()
                outputs.send(sample)
                count += 1
        except Exception, e:
            print e
            time.sleep(1.0)
            continue

if __name__ == "__main__":
    parser = argparse.ArgumentParser("Display info about an input module.")
    parser.add_argument("-i", "--input", default="zpull://*:10000")
    parser.add_argument("-o", "--output", default="zpub://*:10001")
    parser.add_argument("-r", "--report", type=int, default=1000)
    args = parser.parse_args()
    aggregator(args.input, args.output, report=args.report)
