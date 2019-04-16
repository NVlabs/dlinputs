from __future__ import print_function

import os
import sqlite3
from builtins import object, range, str, zip

# Copyright (c) 2017 NVIDIA CORPORATION. All rights reserved.
# See the LICENSE file for licensing terms (BSD-style).


def sqlitedb(dbfile, table="train", epochs=1, cols="*", extra="", verbose=False):
    """Read a dataset from an sqlite3 dbfile and the given table.

    FIXME: update this to standard record conventions, with autoencode/decode
    """
    assert "," not in table
    if "::" in dbfile:
        dbfile, table = dbfile.rsplit("::", 1)
    assert os.path.exists(dbfile)
    sql = "select %s from %s %s" % (cols, table, extra)
    if verbose:
        print("#", sql)
    for epoch in range(epochs):
        if verbose:
            print("# epoch", epoch, "dbfile", dbfile)
        db = sqlite3.connect(dbfile)
        c = db.cursor()
        for row in c.execute(sql):
            cols = [x[0] for x in c.description]
            row = [x for x in row]
            print(row)
            sample = {k: v for k, v in zip(cols, row)}
            sample["__epoch__"] = epoch
            yield sample
        c.close()
        db.close()


sqltypes = None  # FIXME


class SqliteWriter(object):
    """A quick and simple way of writing datasets to sqlite3 files.

    FIXME: update this to standard record conventions, with autoencode/decode
    """

    def __init__(self, fname, tname, **kw):
        self.db = db = sqlite3.connect(fname)
        self.c = c = db.cursor()
        self.keys = keys = list(kw.keys())
        decls = []
        cols = []
        vals = []
        self.converters = {}
        for name in keys:
            kind = kw[name]
            decls += [name + " " + sqltypes.get(kind, "text")]
            cols += [name]
            vals += [":" + name]
            self.converters[name] = converters.get(kind, lambda x: x)
        decls = ",".join(decls)
        cols = ",".join(cols)
        vals = ",".join(vals)
        c.execute("drop table if exists {tname}".format(tname=tname))
        c.execute("create table {tname} ({decls})".format(
            tname=tname, decls=decls))
        self.insert = "insert into {tname} ({cols}) values ({vals})".format(
            tname=tname, cols=cols, vals=vals)

    def add_dict(self, d):
        d = {k: self.converters[k](v) for k, v in list(d.items())}
        self.c.execute(self.insert, d)
        self.db.commit()

    def add(self, **kw):
        self.add_dict(kw)
