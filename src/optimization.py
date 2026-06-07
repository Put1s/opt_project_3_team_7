from collections import defaultdict
import numpy as np
from numpy.linalg import norm, solve
from time import time

def subgradient_method(oracle, x_0, tolerance=1e-5, max_iter=1000, alpha_0=1.0,
                       display=False, trace=False):
    """
    Subgradient descent method for nonsmooth convex optimization.

    Parameters
    ----------
    oracle : BaseNonsmoothConvexOracle-descendant object
        Oracle with .func() and .subgrad() methods implemented for computing
        function value and its one (arbitrary) subgradient respectively.
    x_0 : 1-dimensional np.array
        Starting point of the algorithm
    tolerance : float
        Epsilon value for the stopping criterion:
        ||x_{k+1} - x_k||_2 / max(1, ||x_k||_2) <= tolerance
    max_iter : int
        Maximum number of iterations.
    alpha_0 : float
        Initial value for the sequence of step-sizes (e.g., alpha_k = alpha_0 / sqrt(k+1)).
    display : bool
        If True, debug information is displayed during optimization.
    trace:  bool
        If True, the progress information is appended into history dictionary.

    Returns
    -------
    x_star : np.array
        The point found by the optimization procedure
    message : string
        'success' or 'iterations_exceeded'
    history : dictionary of lists or None
        - history['func'] : list of function values f(x_k)
        - history['time'] : list of floats, containing time in seconds passed from the start
        - history['x'] : list of np.arrays, containing the trajectory (ONLY STORE IF x.size <= 2)
    """
    # TODO: implement.
    pass


def proximal_gradient_method(oracle, x_0, L_0=1.0, tolerance=1e-5,
                             max_iter=1000, trace=False, display=False):
    """
    Proximal Gradient Method (ISTA) for composite optimization.

    Parameters
    ----------
    oracle : BaseCompositeOracle-descendant object
        Oracle with .func(), .grad(), and .prox() methods implemented.
    x_0 : 1-dimensional np.array
        Starting point of the algorithm
    L_0 : float
        Initial value for adaptive line-search (backtracking for Lipschitz constant).
    tolerance : float
        Epsilon value for the stopping criterion:
        ||x_{k+1} - x_k||_2 / max(1, ||x_k||_2) <= tolerance
    max_iter : int
        Maximum number of iterations.
    display : bool
        If True, debug information is displayed during optimization.
    trace:  bool
        If True, the progress information is appended into history dictionary.

    Returns
    -------
    x_star : np.array
        The point found by the optimization procedure
    message : string
        'success' or 'iterations_exceeded'
    history : dictionary of lists or None
    """
    # TODO: Implement
    pass

def proximal_fast_gradient_method(oracle, x_0, L_0=1.0, tolerance=1e-5,
                                  max_iter=1000, trace=False, display=False):
    """
    Fast gradient method (FISTA) for composite minimization.

    Parameters
    ----------
    oracle : BaseCompositeOracle-descendant object
        Oracle with .func(), .grad(), and .prox() methods implemented.
    x_0 : 1-dimensional np.array
        Starting point of the algorithm
    L_0 : float
        Initial value for adaptive line-search (backtracking for Lipschitz constant).
    tolerance : float
        Epsilon value for the stopping criterion:
        ||x_{k+1} - x_k||_2 / max(1, ||x_k||_2) <= tolerance
    max_iter : int
        Maximum number of iterations.
    display : bool
        If True, debug information is displayed during optimization.
    trace:  bool
        If True, the progress information is appended into history dictionary.

    Returns
    -------
    x_star : np.array
        The point found by the optimization procedure
    message : string
        'success' or 'iterations_exceeded'
    history : dictionary of lists or None
        - history['func'] : list of objective function values phi(x_k)
        - history['time'] : list of floats, containing time in seconds passed from the start
    """
    # TODO: Implement
    pass


def frank_wolfe_method(oracle, x_0, R, tolerance=1e-5, max_iter=1000,
                       step_size_strategy='standard', trace=False, display=False):
    """
    Frank-Wolfe (Conditional Gradient) method for constrained optimization:
    min f(x) s.t. ||x||_1 <= R

    Parameters
    ----------
    oracle : BaseSmoothOracle-descendant object
        Oracle with .func() and .grad() methods implemented.
    x_0 : 1-dimensional np.array
        Starting point of the algorithm (usually np.zeros).
    R : float
        Radius of the L1-ball constraint.
    tolerance : float
        Epsilon value for the Frank-Wolfe gap stopping criterion:
        <grad f(x_k), x_k - y_k> <= tolerance
    max_iter : int
        Maximum number of iterations.
    step_size_strategy : str
        'standard' (gamma_k = (k-1)/(k+1)) or 'armijo' (line search).
    display : bool
        If True, debug information is displayed during optimization.
    trace:  bool
        If True, the progress information is appended into history dictionary.

    Returns
    -------
    x_star : np.array
        The point found by the optimization procedure
    message : string
        'success' or 'iterations_exceeded'
    history : dictionary of lists or None
        - history['func'] : list of objective function values f(x_k)
        - history['time'] : list of floats, containing time in seconds passed from the start
        - history['fw_gap'] : list of Frank-Wolfe gaps <grad f(x_k), x_k - y_k>
    """
    # TODO: Implement (Don't forget to use L1RegOracle.lmo to find y_k)
    pass


def barrier_method(oracle, x_0, u_0, lambda_reg, t_0=1.0, mu=10.0,
                   tolerance_inner=1e-6, tolerance_outer=1e-5,
                   max_iter=100, max_inner_iter=100,
                   trace=False, display=False):
    """
    Logarithmic barrier method for L1-regularized optimization.
    min f(x) + lambda * sum(u_i)  s.t.  -u_i <= x_i <= u_i

    Parameters
    ----------
    oracle : BaseSmoothOracle-descendant object
        The SMOOTH oracle for f(x). Must have .func(), .grad(), and .hess().
    x_0 : 1-dimensional np.array
        Starting point for x.
    u_0 : 1-dimensional np.array
        Starting point for u. MUST satisfy strict feasibility: u_0_i > |x_0_i| for all i.
    lambda_reg : float
        L1 regularization coefficient.
    t_0 : float
        Initial value of the barrier parameter.
    mu : float
        Multiplication factor for t on each outer iteration.
    tolerance_inner : float
        Stopping criterion for the inner Newton method (norm of the gradient of F_t).
    tolerance_outer : float
        Stopping criterion for the outer loop: 2 * n / t <= tolerance_outer.
    max_iter : int
        Maximum number of outer iterations.
    max_inner_iter : int
        Maximum number of inner Newton iterations per outer step.
    trace : bool
        If True, the progress information is appended into history dictionary.

    Returns
    -------
    x_star : np.array
        The optimal x.
    u_star : np.array
        The optimal u.
    message : string
        'success' or 'iterations_exceeded'
    history : dictionary of lists or None
    """
    # TODO: Implement (Use BarrierL1Oracle to wrap the base smooth oracle)
    pass