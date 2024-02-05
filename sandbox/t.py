import numpy as np
from scipy.optimize import line_search


def newtons_method(f, grad_f, hess_f, x0, T, c=1 / 8, callback=None):
    """
    Newton's method for optimization with additional improvements.

    Parameters:
    - f: The objective function.
    - grad_f: The gradient of the objective function.
    - hess_f: The Hessian of the objective function.
    - x0: Initial guess.
    - T: Number of iterations.
    - c: Constant used to determine step size.
    - callback: Function called at each iteration for tracking progress.

    Returns:
    - x_T: The optimized solution.
    """
    x_t = np.asarray(x0)

    if x_t.ndim != 1:
        raise ValueError("Initial guess must be a 1-D array.")

    for t in range(T):
        grad_xt = np.asarray(grad_f(x_t))
        hess_xt = np.asarray(hess_f(x_t))

        if grad_xt.shape != x_t.shape or hess_xt.shape != (x_t.shape[0], x_t.shape[0]):
            raise ValueError(
                "Shapes of x_t, grad_f(x_t), hess_f(x_t) are not compatible."
            )

        if not np.all(np.linalg.eigvals(hess_xt) > 0):
            raise ValueError("Hessian should be positive definite at x_t.")

        # Calculate Newton direction
        newton_direction = -np.linalg.solve(hess_xt, grad_xt)

        # Perform a line search along the Newton direction
        step_size = line_search(f, grad_f, x_t, newton_direction)[0]

        # Update step
        x_t = x_t + step_size * newton_direction

        if callback is not None:
            callback(x_t, t)

    return x_t
