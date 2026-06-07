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
    history = defaultdict(list) if trace else None
    x_k = np.array(x_0, dtype=float)
    start = time()

    if trace:
        history['func'].append(oracle.func(x_k))
        history['time'].append(0.0)
        if x_k.size <= 2:
            history['x'].append(x_k.copy())

    message = 'iterations_exceeded'
    for k in range(max_iter):
        g_k = oracle.subgrad(x_k)
        g_norm = norm(g_k)
        if g_norm > 0:
            alpha_k = alpha_0 / np.sqrt(k + 1)
            x_next = x_k - alpha_k * g_k
        else:
            x_next = x_k.copy()

        step_norm = norm(x_next - x_k) / max(1.0, norm(x_k))

        x_k = x_next

        if trace:
            history['func'].append(oracle.func(x_k))
            history['time'].append(time() - start)
            if x_k.size <= 2:
                history['x'].append(x_k.copy())

        if display:
            print(f'iter={k+1}, f={oracle.func(x_k):.6e}, step={step_norm:.3e}')

        if step_norm <= tolerance:
            message = 'success'
            break

    return x_k, message, history


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
    history = defaultdict(list) if trace else None
    x_k = np.array(x_0, dtype=float)
    L = L_0
    start = time()

    if trace:
        history['func'].append(oracle.func(x_k))
        history['time'].append(0.0)
        if x_k.size <= 2:
            history['x'].append(x_k.copy())

    message = 'iterations_exceeded'
    for k in range(max_iter):
        grad_k = oracle.grad(x_k)
        f_k = oracle._f.func(x_k)

        # Backtracking line search for L
        while True:
            x_next = oracle.prox(x_k - grad_k / L, 1.0 / L)
            diff = x_next - x_k
            f_next = oracle._f.func(x_next)
            # Sufficient decrease condition
            if f_next <= f_k + grad_k.dot(diff) + (L / 2.0) * norm(diff) ** 2:
                break
            L *= 2.0

        step_norm = norm(x_next - x_k) / max(1.0, norm(x_k))
        x_k = x_next

        if trace:
            history['func'].append(oracle.func(x_k))
            history['time'].append(time() - start)
            if x_k.size <= 2:
                history['x'].append(x_k.copy())

        if display:
            print(f'iter={k+1}, f={oracle.func(x_k):.6e}, L={L:.3e}, step={step_norm:.3e}')

        if step_norm <= tolerance:
            message = 'success'
            break

    return x_k, message, history


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
    history = defaultdict(list) if trace else None
    x_k = np.array(x_0, dtype=float)
    y_k = x_k.copy()
    t_k = 1.0
    L = L_0
    start = time()

    if trace:
        history['func'].append(oracle.func(x_k))
        history['time'].append(0.0)
        if x_k.size <= 2:
            history['x'].append(x_k.copy())

    message = 'iterations_exceeded'
    for k in range(max_iter):
        grad_k = oracle.grad(y_k)
        f_yk = oracle._f.func(y_k)

        # Backtracking line search for L
        while True:
            x_next = oracle.prox(y_k - grad_k / L, 1.0 / L)
            diff = x_next - y_k
            f_next = oracle._f.func(x_next)
            if f_next <= f_yk + grad_k.dot(diff) + (L / 2.0) * norm(diff) ** 2:
                break
            L *= 2.0

        t_next = (1.0 + np.sqrt(1.0 + 4.0 * t_k ** 2)) / 2.0
        y_next = x_next + (t_k - 1.0) / t_next * (x_next - x_k)

        step_norm = norm(x_next - x_k) / max(1.0, norm(x_k))

        x_k = x_next
        y_k = y_next
        t_k = t_next

        if trace:
            history['func'].append(oracle.func(x_k))
            history['time'].append(time() - start)
            if x_k.size <= 2:
                history['x'].append(x_k.copy())

        if display:
            print(f'iter={k+1}, f={oracle.func(x_k):.6e}, L={L:.3e}, step={step_norm:.3e}')

        if step_norm <= tolerance:
            message = 'success'
            break

    return x_k, message, history


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
        'standard' (gamma_k = 2/(k+2)) or 'armijo' (line search).
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
    history = defaultdict(list) if trace else None
    x_k = np.array(x_0, dtype=float)
    start = time()

    message = 'iterations_exceeded'
    for k in range(max_iter):
        grad_k = oracle.grad(x_k)
        y_k = oracle.lmo(grad_k, R)
        fw_gap = grad_k.dot(x_k - y_k)

        if trace:
            history['func'].append(oracle.func(x_k))
            history['time'].append(time() - start)
            history['fw_gap'].append(fw_gap)

        if display:
            print(f'iter={k+1}, f={oracle.func(x_k):.6e}, fw_gap={fw_gap:.3e}')

        if fw_gap <= tolerance:
            message = 'success'
            break

        d_k = y_k - x_k

        if step_size_strategy == 'standard':
            gamma_k = 2.0 / (k + 2)
        else:  # armijo
            gamma_k = 1.0
            c1 = 1e-4
            f_k = oracle.func(x_k)
            slope = grad_k.dot(d_k)
            for _ in range(30):
                if oracle.func(x_k + gamma_k * d_k) <= f_k + c1 * gamma_k * slope:
                    break
                gamma_k *= 0.5

        x_k = x_k + gamma_k * d_k

    return x_k, message, history


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
    try:
        from src.oracles import BarrierL1Oracle
    except ImportError:
        from oracles import BarrierL1Oracle

    history = defaultdict(list) if trace else None
    x_k = np.array(x_0, dtype=float)
    u_k = np.array(u_0, dtype=float)
    n = len(x_k)
    t = t_0
    start = time()

    if trace:
        history['func'].append(oracle.func(x_k) + lambda_reg * np.sum(u_k))
        history['time'].append(0.0)
        history['t'].append(t)

    message = 'iterations_exceeded'
    for outer in range(max_iter):
        # Check outer stopping criterion
        if 2.0 * n / t <= tolerance_outer:
            message = 'success'
            break

        barrier = BarrierL1Oracle(oracle, lambda_reg, t)
        z_k = np.concatenate([x_k, u_k])

        # Inner Newton loop
        for inner in range(max_inner_iter):
            grad_z = barrier.grad(z_k)
            if norm(grad_z) <= tolerance_inner:
                break

            H_z = barrier.hess(z_k)
            try:
                delta_z = solve(H_z, -grad_z)
            except np.linalg.LinAlgError:
                break

            # Backtracking line search ensuring feasibility
            alpha = 1.0
            x_try, u_try = z_k[:n] + alpha * delta_z[:n], z_k[n:] + alpha * delta_z[n:]
            # Ensure feasibility: u_i > |x_i|
            for _ in range(50):
                x_try = z_k[:n] + alpha * delta_z[:n]
                u_try = z_k[n:] + alpha * delta_z[n:]
                if np.all(u_try - x_try > 0) and np.all(u_try + x_try > 0):
                    # Check sufficient decrease
                    z_try = np.concatenate([x_try, u_try])
                    if barrier.func(z_try) < barrier.func(z_k) + 1e-4 * alpha * grad_z.dot(delta_z):
                        break
                alpha *= 0.5

            z_k = np.concatenate([x_try, u_try])

        x_k = z_k[:n]
        u_k = z_k[n:]
        t *= mu

        if trace:
            history['func'].append(oracle.func(x_k) + lambda_reg * np.sum(u_k))
            history['time'].append(time() - start)
            history['t'].append(t)

        if display:
            obj = oracle.func(x_k) + lambda_reg * np.sum(u_k)
            print(f'outer={outer+1}, t={t:.2e}, obj={obj:.6e}, 2n/t={2*n/t:.3e}')

    return x_k, u_k, message, history
