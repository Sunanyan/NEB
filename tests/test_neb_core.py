
import numpy as np
from neb.neb_core import linear, update


def test_update_shape():
    images = linear([-1., 0.], [1., 0.], 12)
    grad = images.copy()

    updated, fmax = update(images, grad)

    assert updated.shape == images.shape


def test_fixed_endpoints():
    images = linear([-1., 0.], [1., 0.], 12)
    original = images.copy()
    grad = images.copy()

    updated, fmax = update(images, grad)

    np.testing.assert_array_equal(
        updated[0], original[0]
    )

    np.testing.assert_array_equal(
        updated[-1], original[-1]
    )


def test_fmax():
    images = linear([-1., 0.], [1., 0.], 12)
    grad = images.copy()

    updated, fmax = update(images, grad)

    assert np.isfinite(fmax)
    assert fmax >= 0


def test_zero_gradient():
    images = linear([-1., 0.], [1., 0.], 12)
    original = images.copy()
    grad = np.zeros_like(images)

    updated, fmax = update(images, grad)

    np.testing.assert_allclose(
        updated, original, atol=1e-10
    )