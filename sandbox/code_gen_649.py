import numpy as np


def newtons_method(f, grad_f, hess_f, x0, T, c=1/8):
    """
    Newton's method for optimization.

    Parameters:
    - f: The objective function.
    - grad_f: The gradient of the objective function.
    - hess_f: The Hessian of the objective function.
    - x0: Initial guess.
    - T: Number of iterations.
    - c: Constant used to determine step size.

    Returns:
    - x_T: The optimized solution.
    """
    x_t = x0

    for t in range(T):
        grad_xt = grad_f(x_t)
        hess_xt_inv = np.linalg.inv(hess_f(x_t))

        # Computing step size eta
        lambda_xt = np.linalg.norm(grad_xt, ord=2) / np.sqrt(np.dot(grad_xt.T, np.dot(hess_xt_inv, grad_xt)))
        eta = min(c, c / (8 * lambda_xt))

        # Update step
        x_t = x_t - eta * np.dot(hess_xt_inv, grad_xt)

    return x_t

# Example usage
# Define your objective function f, its gradient grad_f, and its Hessian hess_f
# x_optimized = newtons_method(f, grad_f, hess_f, x0, T)