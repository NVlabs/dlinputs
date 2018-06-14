from __future__ import unicode_literals
from dlinputs import utils
import numpy as np

def test_gray():
	assert utils.make_gray(np.zeros((400, 300, 3))).shape == (400, 300)
	assert utils.make_gray(np.zeros((400, 300))).shape == (400, 300)
	assert utils.make_gray(np.zeros((400, 300, 4))).shape == (400, 300)


def test_make_rgb():
	assert utils.make_rgb(np.zeros((400, 300, 3))).shape == (400, 300, 3)
	assert utils.make_rgb(np.zeros((400, 300))).shape == (400, 300, 3)
	assert utils.make_rgb(np.zeros((400, 300, 4))).shape == (400, 300, 3)

def test_make_rgba():
	assert utils.make_rgba(np.zeros((400, 300, 3))).shape == (400, 300, 4)
	assert utils.make_rgba(np.zeros((400, 300))).shape == (400, 300, 4)
	assert utils.make_rgba(np.zeros((400, 300, 4))).shape == (400, 300, 4)

def test_invert_mapping():
	d = dict(a=1, b=2, c=3)
	rd = utils.invert_mapping(d)
	assert rd[1] == "a"

def test_get_string_mapping():
	d = utils.get_string_mapping("a=x:b=y:c=z")
	assert d["a"] == "x"

def test_pilreads():
	image = np.zeros((400, 300, 3))
	png = utils.pildumps(image)
	image1 = utils.pilreads(png, "rgb")
	assert image.shape == image1.shape
	assert (image == image1).all()

def test_pilreads():
	image = np.zeros((400, 300, 3))
	png = utils.pildumps(image)
	image1 = utils.pilreads(png, "gray")
	assert image.shape[:2] == image1.shape

def test_autoencode():
	sample = dict(png=np.zeros((400, 300, 3)))
	raw = utils.autoencode(sample)
	sample1 = utils.autodecode(raw)
	assert (sample["png"] == sample1["png"]).all()


def test_samples_to_batch():
	samples = [dict(png=np.zeros((400, 300, 3)))] * 10
	batch = utils.samples_to_batch(samples)
	assert batch["png"].shape == (10, 400, 300, 3)

	samples = [dict(png=np.zeros((400, x, 3))) for x in [200, 100, 350, 150]]
	batch = utils.samples_to_batch(samples, expand=True)
	assert batch["png"].shape, batch["png"].shape == (4, 400, 350, 3)