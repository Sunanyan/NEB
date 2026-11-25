"""Initial NEB paths: Cartesian interpolation and variational IDPP."""
from dataclasses import dataclass
from numbers import Integral
import numpy as np
from scipy.optimize import minimize

__all__ = ['gen_images', 'linear', 'pair_distances',
           'target_distances', 'idpp_energy_gradient', 'idpp', 'IDPPResult']


def _coords(x):
    x = np.asarray(x, dtype=np.float64)
    if not np.all(np.isfinite(x)):
        raise ValueError('Coordinates must be finite')
    return x


def linear(start, end, n, dtype=np.float64):
    """Return n images including both endpoints; supports arbitrary shapes."""
    if not isinstance(n, Integral) or isinstance(n, bool) or n < 2:
        raise ValueError('n must be an integer >= 2')
    start, end = _coords(start), _coords(end)
    if start.shape != end.shape or start.size == 0:
        raise ValueError('Endpoints must have matching nonempty shapes')
    t = np.linspace(0, 1, n).reshape((n,) + (1,) * start.ndim)
    return ((1-t)*start + t*end).astype(dtype)


def _molecular_coords(x):
    x = _coords(x)
    if x.ndim != 2 or x.shape[0] < 2 or x.shape[1] != 3:
        raise ValueError('IDPP requires coordinates of shape (n_atoms, 3), n_atoms >= 2')
    return x


def pair_distances(coords):
    """Unique pairs in np.triu_indices order."""
    x = _molecular_coords(coords)
    i, j = np.triu_indices(len(x), 1)
    return np.linalg.norm(x[i]-x[j], axis=1)


def target_distances(reactant, product, n_images):
    a, b = _molecular_coords(reactant), _molecular_coords(product)
    if a.shape != b.shape:
        raise ValueError('Endpoint shapes must match')
    if not isinstance(n_images, Integral) or isinstance(n_images, bool) or n_images < 2:
        raise ValueError('n_images must be an integer >= 2')
    s = np.linspace(0, 1, n_images)[:, None]
    return (1-s)*pair_distances(a) + s*pair_distances(b)


def idpp_energy_gradient(coords, target, *, epsilon=1e-4):
    """Return sum (rho-d)^2/rho^4 and Cartesian gradient."""
    x = _molecular_coords(coords)
    i, j = np.triu_indices(len(x), 1)
    d = np.asarray(target, dtype=float)
    if d.shape != i.shape or not np.all(np.isfinite(d)) or np.any(d < 0):
        raise ValueError('Invalid target distances')
    if not np.isfinite(epsilon) or epsilon <= 0:
        raise ValueError('epsilon must be positive and finite')
    delta = x[i]-x[j]
    rho = np.sqrt(np.sum(delta**2, axis=1)+epsilon**2)
    q = rho-d
    energy = np.sum(q**2/rho**4)
    pair_grad = (2*q/rho**5 - 4*q**2/rho**6)[:, None]*delta
    grad = np.zeros_like(x)
    np.add.at(grad, i, pair_grad)
    np.add.at(grad, j, -pair_grad)
    return float(energy), grad


@dataclass
class IDPPResult:
    images: np.ndarray
    success: bool
    message: str
    iterations: int
    objective_history: np.ndarray
    initial_images: np.ndarray
    targets: np.ndarray
    max_gradient: float


def idpp(reactant, product, n_images=9, *, spring=0.1, maxiter=1000,
         gtol=1e-6, epsilon=1e-4, jitter=0.03, seed=42,
         initial_images=None, return_result=False):
    """Variational IDPP with quadratic coupling, optimized by L-BFGS-B.

    This is not the original projected NEB IDPP optimizer. Endpoints stay fixed.
    """
    a, b = _molecular_coords(reactant), _molecular_coords(product)
    x = linear(a, b, n_images)
    targets = target_distances(a, b, n_images)
    if not all(np.isfinite(v) for v in (spring, gtol, epsilon, jitter)) or spring < 0 or gtol <= 0 or epsilon <= 0 or jitter < 0:
        raise ValueError('Invalid IDPP optimization parameters')
    if not isinstance(maxiter, Integral) or maxiter < 1:
        raise ValueError('maxiter must be a positive integer')
    if initial_images is not None:
        guess = _coords(initial_images)
        if guess.shape != x.shape:
            raise ValueError(f'initial_images must have shape {x.shape}')
        x[1:-1] = guess[1:-1]
    if n_images == 2:
        result = IDPPResult(x, True, 'No interior images', 0,
                            np.array([0.5*spring*np.sum(np.diff(x, axis=0)**2)]),
                            x.copy(), targets, 0.)
        return result if return_result else x
    rng = np.random.default_rng(seed)
    if min(pair_distances(z).min() for z in x[1:-1]) < max(0.1, 10*epsilon):
        if jitter == 0:
            raise ValueError('Near collision: supply a noncolliding path or positive jitter')
        noise = rng.normal(size=x.shape[1:])
        noise -= noise.mean(axis=0)
        x[1:-1] += jitter*np.sin(np.pi*np.linspace(0, 1, n_images)[1:-1, None, None])*noise
    start = x.copy()
    shape = x[1:-1].shape

    def objective(z):
        band = np.concatenate((start[:1], z.reshape(shape), start[-1:]), axis=0)
        val = 0.
        grad = np.zeros_like(band)
        for m in range(1, n_images-1):
            v, g = idpp_energy_gradient(band[m], targets[m], epsilon=epsilon)
            val += v
            grad[m] = g
        diff = np.diff(band, axis=0)
        val += 0.5*spring*np.sum(diff**2)
        grad[:-1] -= spring*diff
        grad[1:] += spring*diff
        return val, grad[1:-1].ravel()

    z0 = start[1:-1].ravel()
    history = [objective(z0)[0]]
    opt = minimize(objective, z0, jac=True, method='L-BFGS-B',
                   callback=lambda intermediate_result: history.append(float(intermediate_result.fun)),
                   options={'maxiter': maxiter, 'gtol': gtol, 'ftol': 1e-13, 'maxls': 50})
    x[1:-1] = opt.x.reshape(shape)
    result = IDPPResult(x, bool(opt.success), str(opt.message), int(opt.nit),
                        np.asarray(history), start, targets, float(np.max(np.abs(opt.jac))))
    if return_result:
        return result
    if not result.success:
        raise RuntimeError('IDPP optimization failed: '+result.message)
    return x


def gen_images(reactant, product, n_images, method='linear', **kwargs):
    """Generate a band with linear or IDPP initialization, including endpoints.

    Keyword arguments are forwarded to the chosen method. IDPP returns an
    IDPPResult instead of an array when return_result=True.
    """
    methods = {'linear': linear, 'idpp': idpp}
    key = method.strip().lower()
    if key not in methods:
        raise ValueError(f'Unknown method {method!r}; choose linear or idpp')
    return methods[key](reactant, product, n_images, **kwargs)
