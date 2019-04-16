# Copyright (c) 2017 NVIDIA CORPORATION. All rights reserved.
# See the LICENSE file for licensing terms (BSD-style).

import os
import pickle
import sqlite3


class SqlShelf(object):
    """A shelve-compatible persistent dictionary based on sqlite3"""

    def __init__(self, fname, table="cache", protocol=2, synchronous=False):
        self.fname = fname
        self.table = table
        self.protocol = protocol
        self.db = sqlite3.connect(fname)
        self.c = self.db.cursor()
        self.synchronous = synchronous
        try:
            self.c.execute("select * from {table} limit 1"
                           .format(table=self.table))
        except:
            self.c.execute("create table {table} (key string unique, value blob)"
                           .format(table=self.table))
        self.db.commit()

    def key_map(self, key):
        return str(key)

    def value_map(self, value):
        return pickle.dumps(value, self.protocol)

    def value_unmap(self, value):
        return pickle.loads(value)

    def get(self, key, default=None):
        key = self.key_map(key)
        self.c.execute("select value from {table} where key=?".format(
            table=self.table), (key,))
        for result in self.c:
            return self.value_unmap(result[0])
        return default

    def sync(self):
        self.db.commit()

    def close(self):
        del self.c
        self.db.close()
        del self.db
        self.fname = None

    def itkeys(self):
        it = self.db.cursor()
        it.execute("select key from {table} order by rowid".format(
            table=self.table))
        for row in it:
            yield row[0]
        del it

    def keys(self):
        return list(self.itkeys())

    def itvalues(self):
        it = self.db.cursor()
        it.execute("select value from {table} order by rowid".format(
            table=self.table))
        for row in it:
            yield self.value_unmap(row[0])
        del it

    def values(self):
        return list(self.itvalues())

    def items(self):
        it = self.db.cursor()
        it.execute("select key, value from {table} order by rowid".format(
            table=self.table))
        for row in it:
            yield row[0], self.value_unmap(row[1])
        del it

    def __getitem__(self, key):
        key = self.key_map(key)
        result = self.get(key, self)
        if result == self:
            raise KeyError(key)
        return result

    def __setitem__(self, key, value):
        key = self.key_map(key)
        value = self.value_map(value)
        self.c.execute("insert or replace into {table} values (?, ?)".format(
            table=self.table), (key, memoryview(value)))
        if self.synchronous:
            self.sync()

    def __delitem__(self, key):
        key = self.key_map(key)
        self.c.execute("delete from {table} where key=?".format(
            table=self.table), (key,))
        if self.synchronous:
            self.sync()

    def __contains__(self, key):
        key = self.key_map(key)
        result = self.get(key, self)
        return result != self


def open(*args, **kw):
    return SqlShelf(*args, **kw)
