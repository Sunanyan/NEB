"""Small self-tests for coordinate shapes, gradient formulas, and CI behaviour."""
import numpy as np
from neb.neb_core import linear, neb_force, update
from .neb_fake_potential import *

# Linear interpolation: endpoints, interior, molecule coordinate shape.
a = np.array([[0., 0., 0.], [0., 0., 1.]])
b = np.array([[2., 0., 0.], [0., 0., 3.]])
xyz = linear(a, b, 5)
assert xyz.shape == (5, 2, 3)
assert np.allclose(xyz[2], 0.5 * (a + b))
assert np.array_equal(xyz[0], a) and np.array_equal(xyz[-1], b)

# Given only images + gradient, update works and fixes endpoints.
images = linear([-1., 0.], [1., 0.], 12)
grad = images.copy()
updated, fmax = update(images, grad)
assert updated.shape == images.shape
assert np.array_equal(updated[[0, -1]], images[[0, -1]])
assert fmax >= 0

# In CI, the highest-energy point receives the true force with its tangent
# component reversed, and NO spring force.
three = np.array([[-1.,0.], [0.2,0.], [1.,0.]])
g = np.array([[0.,0.], [2.,3.], [0.,0.]])
e = np.array([0.,1.,0.])
ci_force = neb_force(three, g, e, climb=True)
assert np.allclose(ci_force[1], [2.,-3.])
assert np.allclose(ci_force[[0,-1]], 0)

# Analytic 2D potential gradients versus centered finite differences.
for case in range(1,7):
    x = np.array([-.6,.4]) if case!=6 else np.array([-.7,.8])
    _, g = potential(case, x)
    h=1e-6
    g_fd = np.array([
        (potential(case,x+np.eye(2)[j]*h)[0] -
         potential(case,x-np.eye(2)[j]*h)[0])/(2*h)
        for j in range(2)
    ])
    assert np.allclose(g,g_fd,atol=1e-6,rtol=1e-6), case

print('All tests passed')
