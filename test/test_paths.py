
# coding: utf-8

from dlinputs import paths
from imp import reload

def test_split_sharded_path():
	assert paths.split_sharded_path("http://server/a-@010-b") == ('http://server/a-%03d-b', 10)
	assert paths.split_sharded_path("http://server/a-010-b") == ('http://server/a-010-b', None)

def test_path_shards():
	assert len(list(paths.path_shards("http://server/a-@010-b"))) == 10

def test_find_directory():
	assert paths.find_directory("/bar:/etc:/foo:/bin:/usr:/bam", "lib", verbose=1) == "/usr/lib"

def test_find_file():
	assert paths.find_file("/bar:/etc:/usr/bin:/foo:/bin:/usr:/bam", "ls", verbose=1) == "/bin/ls"

def test_writefile_readfile():
	paths.writefile("/tmp/abc", "def".encode())
	assert paths.readfile("/tmp/abc").decode() == "def"

def test_splitalltext():
	assert paths.splitallext("a/b/c.d.e") == ("a/b/c", "d.e")

#TODO: fix test_find_basenames()
# def test_find_basenames():
# 	basenames = list(paths.find_basenames("testdata", "png"))
# 	assert len(basenames) == 6, len(basenames)

def test_make_save_path():
	reload(paths)
	assert paths.make_save_path("prefix", 5000, 0.1) == 'prefix-000000005-100000.pt'

def test_parse_save_path():
	n, e = paths.parse_save_path('prefix-000000005-100000.pt')
	assert n == 5000
	assert abs(e - 0.1) < 1e-9

# TODO: write test_functions for below
# spliturl
# read_url_path
# findurl
# openurl
# find_url
# read_shards
# extract_shards
# iterate_shards

