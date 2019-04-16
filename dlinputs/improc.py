from __future__ import division

from builtins import range

import numpy as np
import numpy.random as npr
import scipy.ndimage as ndi
from numpy import cos, sin
from past.utils import old_div

# Copyright (c) 2017 NVIDIA CORPORATION. All rights reserved.
# See the LICENSE file for licensing terms (BSD-style).


def invert(image):
    """Invert the given image.

    :param image: image
    :returns: inverted image

    """
    assert np.amin(image) >= -1e-6
    assert np.amax(image) <= 1+1e-6
    return 1.0 - np.clip(image, 0, 1.0)


def autoinvert(image):
    """Autoinvert the given document image.

    If the image appears to be black on white, invert it to white on black,
    otherwise leave it unchanged.

    :param image: document image
    :returns: autoinverted document image

    """
    assert np.amin(image) >= -1e-6
    assert np.amax(image) <= 1+1e-6
    if np.median(image) > 0.5:
        return 1.0 - image
    else:
        return image


def make_distortions(size, distortions=[(5.0, 3)]):
    """Generate 2D distortions using filtered Gaussian noise.

    The distortions are a sum of gaussian filtered white noise
    with the given sigmas and maximum distortions.

    :param size: size of the image for which distortions are generated
    :param distortions: list of (sigma, maxdist) pairs
    :returns: a grid of source coordinates suitable for scipy.ndimage.map_coordinates

    """
    import pylab  # FIXME
    h, w = size
    total = np.zeros((2, h, w), 'f')
    for sigma, maxdist in distortions:
        deltas = pylab.randn(2, h, w)
        deltas = ndi.gaussian_filter(deltas, (0, sigma, 0))
        deltas = ndi.gaussian_filter(deltas, (0, 0, sigma))
        r = np.amax((deltas[..., 0]**2 + deltas[..., 1]**2)**.5)
        deltas *= old_div(maxdist, r)
        total += deltas
    deltas = total
    xy = np.array(np.meshgrid(list(range(h)), list(range(w)))
                  ).transpose(0, 2, 1)
    coords = deltas + xy
    return coords


def map_image_coordinates(image, coords, order=1, mode="nearest"):
    """Given an image, map the image coordinates for each channel.

    :param image: rank 2 or 3 image
    :param coords: coords to map to
    :param order: order of the interpolation
    :param mode: mode for the boundary
    :returns: distorted image

    """
    if image.ndim == 2:
        return ndi.map_coordinates(image, coords, order=order)
    elif image.ndim == 3:
        result = np.zeros(image.shape, image.dtype)
        for i in range(image.shape[-1]):
            ndi.map_coordinates(
                image[..., i], coords, order=order, output=result[..., i], mode=mode)
        return result


def random_distortions(images, distortions=[(5.0, 3)], order=1, mode="nearest"):
    """Apply a random distortion to a list of images.

    All images must have the same width and height.

    :param images: list of images
    :param distortions: list of distortion parameters for `make_distortions`
    :param order: order of the interpolation
    :param mode: boundary handling
    :returns: list of distorted images

    """
    h, w = images[0].shape[:2]
    coords = make_distortions((h, w), distortions)
    return [map_image_coordinates(image, coords, order=order, mode=mode) for image in images]

def random_distortion(image, distortions=[(5.0, 3)], order=1, mode="nearest"):
    """Apply a random distortion to an image.

    :param image: image
    :param distortions: list of distortion parameters for `make_distortions`
    :param order: order of the interpolation
    :param mode: boundary handling
    :returns: distorted image

    """
    h, w = image.shape[:2]
    coords = make_distortions((h, w), distortions)
    return map_image_coordinates(image, coords, order=order, mode=mode)


def random_scaled_rotation(ralpha=(-0.2, 0.2), rscale=((0.8, 1.2), (0.8, 1.2))):
    """Compute a random transformation matrix for a scaled rotation.

    :param ralpha: range of rotation angles
    :param rscale: range of scales for x and y
    :returns: random transformation

    """
    affine = np.eye(2)
    if rscale is not None:
        (x0, x1), (y0, y1) = rscale
        affine = np.diag([npr.uniform(x0, x1), npr.uniform(y0, y1)])
    if ralpha is not None:
        a0, a1 = ralpha
        a = npr.uniform(a0, a1)
        c = cos(a)
        s = sin(a)
        m = np.array([[c, -s], [s, c]], 'f')
        affine = np.dot(m, affine)
    return affine


def random_gamma(image, rgamma=(0.5, 2.0), cgamma=(0.8, 1.2)):
    """Perform a random gamma transformation on an image.

    :param image: input image
    :param rgamma: grayscale gamma range
    :param cgamma: separate per channel color gamma range
    :returns: transformed image

    """
    image = image.copy()
    if rgamma is not None:
        gamma = npr.uniform(*rgamma)
    else:
        gamma = 1.0
    for plane in range(3):
        g = gamma
        if cgamma is not None:
            g *= npr.uniform(*cgamma)
        image[..., plane] = image[..., plane] ** g
    return image


def standardize(image, size, crop=0, mode="nearest", affine=np.eye(2)):
    """Rescale and crop the image to the given size.

    With crop=1, this rescales the image so that the target size fits
    snugly into it and cuts out the center; with crop=0, this rescales
    the image so that the image fits into the target size and fills
    the boundary in according to `mode`.

    :param ndarray image: image to be standardized
    :param tuple size: target size
    :param bool crop: crop the image
    :param str mode: boundary mode
    :param affine: affine transformation to be applied
    :returns: standardized image
    :rtype: ndarray

    """
    h, w = image.shape[:2]
    th, tw = size
    oshape = (th, tw, image.shape[2])
    if crop:
        scale = max(h * 1.0 / th, w * 1.0 / tw)
    else:
        scale = min(h * 1.0 / th, w * 1.0 / tw)
    affine = np.eye(2)
    affine = affine * scale
    center = old_div(np.array(image.shape[:2], 'f'), 2)
    tcenter = old_div(np.array([th, tw], 'f'), 2)
    delta = np.matmul(affine, tcenter) - center
    matrix = np.eye(3)
    matrix[:2, :2] = affine
    offset = np.zeros(3)
    offset[:2] = -delta
    result = ndi.affine_transform(image, matrix, offset, order=1,
                                  output_shape=oshape, mode=mode)
    return result


def pilproc(*args, **kw):
    """Apply PIL image processing to NumPy arrays."""
    def f(image, planar = kw.get("planar", False), otype = kw.get("dtype", None)):
        import PIL
        assert image.dtype in [np.uint8, np.float16, np.float32, np.float64], image.dtype
        itype = image.dtype
        if planar: image = image.transpose(1, 2, 0)
        if image.dtype != np.uint8:
            assert np.amin(image)>=0 and np.amax(image)<=1
            image = (255*image).astype(np.uint8)
        img = PIL.Image.fromarray(image)
        for f in args: img = f(img)
        otype = otype or itype
        if otype != np.uint8:
            output = np.asarray(img, otype) / 255.0
        else:
            output = np.asarray(img, otype)
        if planar: output = output.transpose(2, 0, 1)
        return output
    return f
