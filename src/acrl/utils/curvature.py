# Adapted from github gist https://gist.github.com/elyase/451cbc00152cb99feac6
import numpy as np
from scipy.interpolate import UnivariateSpline


def curvature_splines(x: np.array, y: np.array, error: float = 0.1) -> np.array:
    """Calculate the signed curvature of a 2D curve at each point
    using interpolating splines.
    Parameters
    ----------
    x,y: numpy.array(dtype=float) shape (n_points, )
         or
         y=None and
         x is a numpy.array(dtype=complex) shape (n_points, )
         In the second case the curve is represented as a np.array
         of complex numbers.
    error : float
        The admisible error when interpolating the splines
    Returns
    -------
    curvature: numpy.array shape (n_points)
    Note: This is 2-3x slower (1.8 ms for 2000 points) than `curvature_gradient`
    but more accurate, especially at the borders.
    """
    t = np.arange(x.shape[0])
    std = error * np.ones_like(x)

    fx = UnivariateSpline(t, x, k=4, w=1 / np.sqrt(std))
    fy = UnivariateSpline(t, y, k=4, w=1 / np.sqrt(std))

    dx = fx.derivative(1)(t)
    ddx = fx.derivative(2)(t)
    dy = fy.derivative(1)(t)
    ddy = fy.derivative(2)(t)
    curvature = (dx * ddy - dy * ddx) / np.power(dx**2 + dy**2, 3.0 / 2.0)
    return curvature.reshape(-1, 1)
