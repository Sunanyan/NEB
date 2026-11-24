import numpy as np
from neb.neb_core import linear


def test_shape():
    a = np.zeros((6, 3))
    b = np.ones((6, 3))

    images = linear(a, b, 9)

    assert images.shape == (9, 6, 3)


def test_endpoints():
    a = np.array([0., 0.])
    b = np.array([2., 4.])

    images = linear(a, b, 9)

    np.testing.assert_allclose(images[0], a)
    np.testing.assert_allclose(images[-1], b)


def test_midpoint():
    a = np.array([0., 0.])
    b = np.array([2., 4.])

    images = linear(a, b, 9)

    np.testing.assert_allclose(
        images[4], (a + b) / 2
    )


def test_equal_spacing():
    a = np.array([0., 0.])
    b = np.array([2., 4.])

    images = linear(a, b, 9)
    d = np.diff(images, axis=0)

    np.testing.assert_allclose(
        d,
        np.broadcast_to(d[0], d.shape)
    )