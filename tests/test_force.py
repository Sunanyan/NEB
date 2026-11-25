import numpy as np
import pytest

from neb.neb_core import neb_force


METHODS = [
    "central",
    "bisector",
    "upwind",
    "improved",
]


@pytest.fixture
def band():
    images = np.array([
        [0., 0.],
        [1., 0.],
        [2., 0.]
    ])

    energy = np.array([0., 1., 2.])

    return images, energy


# =========================================================
# Fixed endpoints
# =========================================================

@pytest.mark.parametrize("method", METHODS)
def test_fixed_endpoints(method, band):

    x, e = band
    grad = np.ones_like(x)

    f = neb_force(
        x, grad, e,
        tangent=method
    )

    np.testing.assert_allclose(f[0], 0.)
    np.testing.assert_allclose(f[-1], 0.)


# =========================================================
# Parallel gradient should be removed
# =========================================================

@pytest.mark.parametrize("method", METHODS)
def test_parallel_gradient(method, band):

    x, e = band

    grad = np.array([
        [0., 0.],
        [5., 0.],
        [0., 0.]
    ])

    f = neb_force(
        x, grad, e,
        tangent=method
    )

    np.testing.assert_allclose(
        f[1], [0., 0.],
        atol=1e-10
    )


# =========================================================
# Perpendicular gradient should be preserved
# =========================================================

@pytest.mark.parametrize("method", METHODS)
def test_perpendicular_gradient(method, band):

    x, e = band

    grad = np.array([
        [0., 0.],
        [0., 5.],
        [0., 0.]
    ])

    f = neb_force(
        x, grad, e,
        tangent=method
    )

    np.testing.assert_allclose(
        f[1], [0., -5.],
        atol=1e-10
    )


# =========================================================
# Spring force
# =========================================================

@pytest.mark.parametrize("method", METHODS)
def test_spring_force(method):

    x = np.array([
        [0., 0.],
        [1., 0.],
        [3., 0.]
    ])

    e = np.array([0., 1., 2.])
    grad = np.zeros_like(x)

    f = neb_force(
        x, grad, e,
        k=2.,
        tangent=method
    )

    np.testing.assert_allclose(
        f[1], [2., 0.],
        atol=1e-10
    )


# =========================================================
# Combined force
# =========================================================

@pytest.mark.parametrize("method", METHODS)
def test_combined_force(method):

    x = np.array([
        [0., 0.],
        [1., 0.],
        [3., 0.]
    ])

    e = np.array([0., 1., 2.])

    grad = np.array([
        [0., 0.],
        [3., 4.],
        [0., 0.]
    ])

    f = neb_force(
        x, grad, e,
        k=2.,
        tangent=method
    )

    np.testing.assert_allclose(
        f[1], [2., -4.],
        atol=1e-10
    )


# =========================================================
# Climbing image
# =========================================================

@pytest.mark.parametrize("method", METHODS)
def test_climbing_image(method):

    x = np.array([
        [0., 0.],
        [1., 0.],
        [3., 0.]
    ])

    e = np.array([0., 2., 0.])

    grad = np.array([
        [0., 0.],
        [3., 4.],
        [0., 0.]
    ])

    f = neb_force(
        x, grad, e,
        k=2.,
        climb=True,
        tangent=method
    )

    np.testing.assert_allclose(
        f[1], [3., -4.],
        atol=1e-10
    )


# =========================================================
# Spring constant scaling
# =========================================================

@pytest.mark.parametrize("method", METHODS)
def test_spring_scaling(method):

    x = np.array([
        [0., 0.],
        [1., 0.],
        [3., 0.]
    ])

    e = np.array([0., 1., 2.])
    grad = np.zeros_like(x)

    f1 = neb_force(
        x, grad, e,
        k=1.,
        tangent=method
    )

    f2 = neb_force(
        x, grad, e,
        k=5.,
        tangent=method
    )

    np.testing.assert_allclose(
        f2, 5*f1,
        atol=1e-10
    )


# =========================================================
# Regression: missing energy
# =========================================================

def test_without_energy():

    x = np.array([
        [0., 0.],
        [1., 0.],
        [2., 0.]
    ])

    grad = np.zeros_like(x)

    f = neb_force(x, grad)

    assert np.all(np.isfinite(f))

    np.testing.assert_allclose(
        f, 0.,
        atol=1e-10
    )