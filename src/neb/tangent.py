import numpy as np


__all__ = [
    "tangent_central",
    "tangent_bisector",
    "tangent_upwind",
    "tangent_improved",
    "compute_tangents",
    "TANGENT_METHODS",
]

EPS = 1e-14


def normalize(v, eps=EPS):
    """Return a unit vector."""
    norm = np.linalg.norm(v)

    if norm < eps:
        return np.zeros_like(v)

    return v / norm



def tangent_central(x, energy, i):
    """Central difference tangent."""
    return x[i + 1] - x[i - 1]



def tangent_bisector(x, energy, i):
    """Normalized geometric bisector tangent."""
    a = x[i] - x[i - 1]
    b = x[i + 1] - x[i]

    return normalize(a) + normalize(b)



def tangent_upwind(x, energy, i):
    """Energy-based one-sided tangent."""

    a = x[i] - x[i - 1]
    b = x[i + 1] - x[i]

    if energy[i + 1] > energy[i - 1]:
        return b

    elif energy[i + 1] < energy[i - 1]:
        return a

    return a + b



def tangent_improved(x, energy, i):
    """Henkelman-Jonsson improved energy-weighted tangent."""

    a = x[i] - x[i - 1]
    b = x[i + 1] - x[i]

    em, ei, ep = energy[i - 1:i + 2]

    if ep > ei > em:
        return b

    if ep < ei < em:
        return a

    # Local energy extrema
    de_a = abs(ei - em)
    de_b = abs(ep - ei)

    de_max = max(de_a, de_b)
    de_min = min(de_a, de_b)

    if ep > em:
        return b * de_max + a * de_min

    elif ep < em:
        return b * de_min + a * de_max

    return a + b



TANGENT_METHODS = {
    "central": tangent_central,
    "bisector": tangent_bisector,
    "upwind": tangent_upwind,
    "improved": tangent_improved,
}



def compute_tangents(images, energy=None, method="improved", eps=EPS):
    """
    Compute normalized tangents for all internal images.

    Parameters
    ----------
    images : ndarray
        Image configurations, shape (M, ...).

    energy : ndarray or None
        Image energies, shape (M,).

    method : str or callable
        Tangent estimation method:
        "central", "bisector", "upwind", "improved",
        or a custom tangent function.

    eps : float
        Numerical tolerance for degenerate tangents.

    Returns
    -------
    tau : ndarray
        Unit tangents with the same shape as images.
        Endpoint tangents are zero.
    """
    if method in ("upwind", "improved") and energy is None:
        raise ValueError(
            f"Tangent method '{method}' requires energy."
        )
    
    x = np.asarray(images)
    shape = x.shape

    n = len(x)

    x = x.reshape(n, -1)
    e = None if energy is None else np.asarray(energy)

    if callable(method):
        tangent_func = method

    elif method in TANGENT_METHODS:
        tangent_func = TANGENT_METHODS[method]

    tau = np.zeros_like(x)

    # Compute internal tangents
    for i in range(1, n - 1):

        t = tangent_func(x, e, i)
        norm = np.linalg.norm(t)

        # Fall back to central difference
        if norm < eps:
            t = tangent_central(x, e, i)
            norm = np.linalg.norm(t)

        tau[i] = t / norm

    return tau.reshape(shape)