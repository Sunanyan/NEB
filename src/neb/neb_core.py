import numpy as np
from .tangent import *



def neb_force(images,
              grad,
              energy=None,
              k=2.0,
              climb=False,
              tangent="improved"
        ):
    """NEB forces, using an energy-weighted tangent and optional climbing image.

    images/grad: (number of images, ...) array.
    energy: optional (number of images,) array for improved tangent; required for CI.
    endpoints receive exactly zero NEB force.
    """
    shape = np.shape(images)
    x = np.asarray(images)
    g = np.asarray(grad)
    e = None if energy is None else np.asarray(energy)
    
    x = x.reshape(len(x), -1)
    g = g.reshape(len(g), -1)
    f = np.zeros_like(x)
    ci = 1 + np.argmax(e[1:-1]) if climb else -1

    if e is None:
        if climb:
            raise ValueError("Climbing image requires energy.")

        if tangent in ("upwind", "improved"):
            tangent = "central"

    tangents = compute_tangents(x, e, method=tangent)

    for i in range(1, len(x) - 1):
        a = x[i] - x[i - 1]
        b = x[i + 1] - x[i]

        tau = tangents[i]

        true_force = -g[i]
        parallel = np.dot(true_force, tau) * tau

        if i == ci:
            f[i] = true_force - 2 * parallel
        else:
            spring = k * (np.linalg.norm(b) - np.linalg.norm(a)) * tau
            f[i] = true_force - parallel + spring

    return f.reshape(shape)



def update(images, grad, energy=None, step=0.02, k=2., climb=False, max_move=0.04, dtype=np.float32):
    """One NEB or CI-NEB iteration, return (updated images, maximum NEB force)."""
    f = neb_force(images, grad, energy, k=k, climb=climb)
    n = len(images)
    f_flat = f.reshape(n, -1)
    fmax = np.linalg.norm(f_flat[1:-1], axis=1).max()
    move = step * f_flat
    if max_move is not None:
        lengths = np.linalg.norm(move, axis=1)
        move *= np.minimum(1, max_move / np.maximum(lengths, 1e-15))[:, None]
    result = np.asarray(images, dtype=dtype).copy()
    result[1:-1] += move.reshape(result.shape)[1:-1]
    return result, fmax



def evaluate_band(
    images,
    energy_gradient,
    fixed_endpoint_cache=None,
    progress_callback=None
):
    """Evaluate energies and gradients for all NEB images."""
    images = np.asarray(images)

    energies = np.empty(len(images))
    gradients = np.zeros_like(images)

    for i, coords in enumerate(images):
        key = -1 if i == len(images) - 1 else i

        if (
            fixed_endpoint_cache is not None
            and key in fixed_endpoint_cache
        ):
            energies[i] = fixed_endpoint_cache[key]

        else:
            energies[i], gradients[i] = energy_gradient(coords)

        if progress_callback is not None:
            progress_callback(i)

    return energies, gradients


