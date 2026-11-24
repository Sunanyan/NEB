import numpy as np

def linear(start, end, n, dtype = np.float32):
    """Return n images including the two endpoints."""
    start = np.asarray(start, dtype=dtype)
    end   = np.asarray(end, dtype=dtype)
    t = np.linspace(0, 1, n)
    t = t.reshape((n,) + (1,) * start.ndim)
    return (1 - t) * start + t * end


def neb_force(images, grad, energy=None, k=2., climb=False, dtype=np.float32):
    """NEB forces, using an energy-weighted tangent and optional climbing image.

    images/grad: (number of images, ...) array.
    energy: optional (number of images,) array for improved tangent; required for CI.
    endpoints receive exactly zero NEB force.
    """
    shape = np.shape(images)
    x = np.asarray(images, dtype=dtype)
    g = np.asarray(grad, dtype=dtype)
    e = None if energy is None else np.asarray(energy, dtype=dtype)
    if climb and e is None:
        raise ValueError("CI needs energies to select the highest image")
    
    x = x.reshape(len(x), -1)
    g = g.reshape(len(g), -1)
    f = np.zeros_like(x)
    ci = 1 + np.argmax(e[1:-1]) if climb else -1

    for i in range(1, len(x) - 1):
        a = x[i] - x[i - 1]  
        b = x[i + 1] - x[i]  
        if e is not None:
            de_a = abs(e[i] - e[i - 1])
            de_b = abs(e[i + 1] - e[i])

        # Without energies use a geometric tangent; otherwise an improved tangent.
        if e is None:
            tau = a + b
        elif e[i + 1] > e[i] > e[i - 1]:
            tau = b.copy()
        elif e[i + 1] < e[i] < e[i - 1]:
            tau = a.copy()
        elif e[i + 1] > e[i - 1]:
            tau = b * max(de_a, de_b) + a * min(de_a, de_b)
        elif e[i + 1] < e[i - 1]:
            tau = b * min(de_a, de_b) + a * max(de_a, de_b)
        else:
            tau = a + b

        norm = np.linalg.norm(tau)
        if norm < 1e-14:
            tau = a + b
            norm = np.linalg.norm(tau)
        if norm < 1e-14:
            continue
        tau /= norm

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
    f = neb_force(images, grad, energy, k=k, climb=climb, dtype=dtype)
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


