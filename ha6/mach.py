import numpy as np

def func_f(x):
    return x**3 - 2*x + 2 # -1.76929235423863

def deri_f(x):
    return 3 * x**2 - 2

def func_g(x):
    return 6*x/(x**2 + 1)

def deri_g(x):
    return 6 * (1 - x**2) / (x**2 + 1)**2

def find_root_newton(f: object, df: object, start: np.inexact, n_iters_max: int = 60) -> (np.inexact, int):
    """
    Find a root of f(x)/f(z) starting from start using Newton's method.

    Arguments:
    f: function object (assumed to be continuous), returns function value if called as f(x)
    df: derivative of function f, also callable
    start: start position, can be either float (for real valued functions) or complex (for complex valued functions)
    n_iters_max: maximum number of iterations (optional)

    Return:
    root: approximate root, should have the same format as the start value start
    n_iterations: number of iterations
    """

    assert(n_iters_max > 0)

    # Initialize root with start value
    root = start

    # chose meaningful convergence criterion eps, e.g 10 * eps
    convergence_criterion = 10 * np.finfo(np.float64).eps

    # Initialize iteration
    fc = f(root)
    dfc = df(root)
    n_iterations = 0

    # TODO: loop until convergence criterion eps is met
    for _ in range(n_iters_max):
        print(root)
        # return root and n_iters_max+1 if abs(derivative) is below f_eps or abs(root) is above 1e5 (to avoid divergence)
        if abs(dfc) < convergence_criterion or abs(root) > 1e5:
            return root, n_iters_max+1
        # update root value and function/dfunction values
        root = root - fc / dfc
        fc = f(root)
        dfc = df(root)
        # avoid infinite loops and return (root, n_iters_max+1)
        n_iterations += 1

    return root, n_iterations

print(find_root_newton(func_g, deri_g, 2, 60))