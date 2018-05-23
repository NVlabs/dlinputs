from dlinputs import tarrecords
reload(tarrecords)
import numpy as np
import glob

def test_tardata():
	stream = open("testdata/imagenet-000000.tgz")
	data = tarrecords.tardata(stream)
	samples = list(data)
	assert samples[0] == ('10.cls', '304'), samples[0]
	assert {2} == set([len(x) for x in samples])

def test_group_by_keys():
	stream = open("testdata/imagenet-000000.tgz")
	data = tarrecords.tardata(stream)
	data = tarrecords.group_by_keys()(data)
	samples = list(data)
	keys = samples[0].keys()
	assert 'png' in keys
	assert 'cls' in keys

# get_ipython().system(u'tar -ztvf testdata/imagenet-000000.tgz | sed 7q')
# get_ipython().system(u'tar xvf testdata/imagenet-000000.tgz 10.png')
# get_ipython().system(u'file 10.png')

def test_decoder():
	stream = open("testdata/imagenet-000000.tgz")
	data = tarrecords.tardata(stream)
	data = tarrecords.group_by_keys()(data)
	data = tarrecords.decoder()(data)
	samples = list(data)
	# print samples[0].keys()
	keys = samples[0].keys()
	assert 'png' in keys
	assert 'cls' in keys

def test_tariterator1():
	stream = open("testdata/imagenet-000000.tgz")
	data = tarrecords.tariterator1(stream)
	samples = list(data)
	assert len(samples)==47
	assert samples[0]["__key__"] == "10", samples[0]["__key__"]
	assert set(samples[3].keys()) == set("__key__ png cls xml wnid".split()), samples[3].keys()
	assert samples[-1]["png"].shape == (400, 300, 3)

def test_tariterator():
	stream = open("testdata/imagenet-000000.tgz")
	data = tarrecords.tariterator(stream)
	samples = list(data)
	assert len(samples)==47
	for i in xrange(len(samples)):
	    assert samples[i]["png"].dtype == np.dtype('f'), samples[i]["png"].dtype
	    assert np.amin(samples[i]["png"]) >= 0, np.amin(samples[i]["png"])
	    assert np.amin(samples[i]["png"]) <= 1, np.amax(samples[i]["png"])
	assert samples[0]["__key__"] == "10"
	assert set(samples[3].keys()) == set("__key__ __source__ cls png xml wnid".split()), samples[3].keys()
	assert samples[-1]["png"].shape == (400, 300, 3)

def test_TarWriter():
	stream = open("testdata/imagenet-000000.tgz")
	data = tarrecords.tariterator(stream)
	samples = list(data)
		
	stream = open("/tmp/test.tgz", "wb")
	sink = tarrecords.TarWriter(stream)
	for sample in samples:
	    sink.write(sample)
	sink.close()
	stream.close()

	# Check if test.tgz was created
	assert len(glob.glob("/tmp/test.tgz")) == 1

	stream = open("/tmp/test.tgz")
	data = tarrecords.tariterator(stream)
	samples = list(data)
	assert len(samples)==47
	assert samples[0]["__key__"] == "10"
	assert set(samples[3].keys()) == set("__key__ __source__ cls png xml wnid".split()), samples[3].keys()
	assert samples[-1]["png"].shape == (400, 300, 3)

