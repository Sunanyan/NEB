import numpy as np
import pytest

from neb.tangent import compute_tangents


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
        [1., 1.],
        [2., 1.],
        [3., 0.]
    ])

    energy = np.array([0., 1., 2., 0.])

    return images, energy


@pytest.mark.parametrize("method", METHODS)
def test_shape(method, band):
    x, e = band

    tau = compute_tangents(x, e, method=method)

    assert tau.shape == x.shape


@pytest.mark.parametrize("method", METHODS)
def test_normalization(method, band):
    x, e = band

    tau = compute_tangents(x, e, method=method)

    np.testing.assert_allclose(
        np.linalg.norm(tau[1:-1], axis=1),
        1.,
        atol=1e-10
    )


@pytest.mark.parametrize("method", METHODS)
def test_straight_line(method):
    x = np.array([
        [0., 0.],
        [1., 0.],
        [3., 0.],
        [6., 0.]
    ])

    e = np.array([0., 1., 2., 3.])

    tau = compute_tangents(x, e, method=method)

    np.testing.assert_allclose(
        tau[1:-1],
        [[1., 0.], [1., 0.]],
        atol=1e-10
    )


@pytest.mark.parametrize("method", METHODS)
def test_translation_invariance(method, band):
    x, e = band

    t1 = compute_tangents(x, e, method=method)
    t2 = compute_tangents(x + 10., e, method=method)

    np.testing.assert_allclose(t1, t2, atol=1e-10)


@pytest.mark.parametrize("method", METHODS)
def test_scaling_invariance(method, band):
    x, e = band

    t1 = compute_tangents(x, e, method=method)
    t2 = compute_tangents(5*x, e, method=method)

    np.testing.assert_allclose(t1, t2, atol=1e-10)


@pytest.mark.parametrize("method", METHODS)
def test_rotation_equivariance(method, band):
    x, e = band

    theta = np.pi / 3

    Q = np.array([
        [np.cos(theta), -np.sin(theta)],
        [np.sin(theta),  np.cos(theta)]
    ])

    t1 = compute_tangents(x, e, method=method)
    t2 = compute_tangents(x @ Q.T, e, method=method)

    np.testing.assert_allclose(
        t2,
        t1 @ Q.T,
        atol=1e-10
    )


@pytest.mark.parametrize("method", METHODS)
def test_energy_shift_invariance(method, band):
    x, e = band

    t1 = compute_tangents(x, e, method=method)
    t2 = compute_tangents(x, e + 100., method=method)

    np.testing.assert_allclose(t1, t2, atol=1e-10)


@pytest.mark.parametrize("method", METHODS)
def test_finite_values(method, band):
    x, e = band

    tau = compute_tangents(x, e, method=method)

    assert np.all(np.isfinite(tau))