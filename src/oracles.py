import numpy as np
import scipy
from scipy.special import expit


class BaseSmoothOracle(object):
    """
    Base class for implementation of oracles.
    """

    def func(self, x):
        """
        Computes the value of function at point x.
        """
        raise NotImplementedError('Func oracle is not implemented.')

    def grad(self, x):
        """
        Computes the gradient at point x.
        """
        raise NotImplementedError('Grad oracle is not implemented.')

    def hess(self, x):
        """
        Computes the Hessian matrix at point x.
        """
        raise NotImplementedError('Hessian oracle is not implemented.')

    def func_directional(self, x, d, alpha):
        """
        Computes phi(alpha) = f(x + alpha*d).
        """
        return np.squeeze(self.func(x + alpha * d))

    def grad_directional(self, x, d, alpha):
        """
        Computes phi'(alpha) = (f(x + alpha*d))'_{alpha}
        """
        return np.squeeze(self.grad(x + alpha * d).dot(d))


class QuadraticOracle(BaseSmoothOracle):
    """
    Oracle for quadratic function:
       func(x) = 1/2 x^TAx - b^Tx.
    """

    def __init__(self, A, b):
        if scipy.sparse.issparse(A):
            if (A != A.T).nnz != 0:
                raise ValueError('A should be a symmetric matrix.')
        else:
            if not np.allclose(A, A.T):
                raise ValueError('A should be a symmetric matrix.')

        self.A = A
        self.b = b

    def func(self, x):
        return 0.5 * np.dot(self.A.dot(x), x) - self.b.dot(x)

    def grad(self, x):
        return self.A.dot(x) - self.b

    def hess(self, x):
        return self.A


class NonConvexOracle(BaseSmoothOracle):
    """
    Oracle for test function from your assignment.
    """

    def __init__(self):
        pass

    def func(self, x):
        x = np.asarray(x, dtype='d')
        return (x[0] ** 2 + x[1] - 11) ** 2 + (x[0] + x[1] ** 2 - 7) ** 2

    def grad(self, x):
        x = np.asarray(x, dtype='d')
        return np.array([4 * x[0] * (x[0] ** 2 + x[1] - 11) + 2 * (x[0] + x[1] ** 2 - 7),
                         2 * (x[0] ** 2 + x[1] - 11) + 4 * x[1] * (x[0] + x[1] ** 2 - 7)], dtype='d')

    def hess(self, x):
        x = np.asarray(x, dtype='d')
        return np.array([
            [12 * x[0] ** 2 + 4 * x[1] - 42, 4 * x[0] + 4 * x[1]],
            [4 * x[0] + 4 * x[1], 12 * x[1] ** 2 + 4 * x[0] - 26]
        ], dtype='d')


class PoissonL2Oracle(BaseSmoothOracle):
    """
    Oracle for regression loss  function with l2 regularization:
         check your individual assignment

    Let A and b be parameters of the model (feature matrix
    and labels vector respectively).

    Parameters
    ----------
        matvec_Ax : function
            Computes matrix-vector product Ax, where x is a vector of size n.
        matvec_ATx : function of x
            Computes matrix-vector product A^Tx, where x is a vector of size m.
        matmat_ATsA : function
            Computes matrix-matrix-matrix product A^T * Diag(s) * A,
    """

    def __init__(self, matvec_Ax, matvec_ATx, matmat_ATsA, b, regcoef):
        self.matvec_Ax = matvec_Ax
        self.matvec_ATx = matvec_ATx
        self.matmat_ATsA = matmat_ATsA
        self.b = b
        self.regcoef = regcoef
        self.m = b.size

    def func(self, x):
        z = np.asarray(self.matvec_Ax(x)).ravel()
        z = np.clip(z, -50, 50)
        return np.mean(np.exp(z) - self.b * z) + 0.5 * self.regcoef * np.dot(x, x)

    def grad(self, x):
        z = np.asarray(self.matvec_Ax(x)).ravel()
        z = np.clip(z, -50, 50)
        return np.asarray(self.matvec_ATx(np.exp(z) - self.b)) / self.m + self.regcoef * x

    def hess(self, x):
        z = np.asarray(self.matvec_Ax(x)).ravel()
        z = np.clip(z, -50, 50)
        return np.asarray(self.matmat_ATsA(np.exp(z))) / self.m + self.regcoef * np.eye(x.size)


class LogisticL2Oracle(BaseSmoothOracle):
    """
    Oracle for classification loss  function with l2 regularization:
         check your individual assignment

    Let A and b be parameters of the model (feature matrix
    and labels vector respectively).

    Parameters
    ----------
        matvec_Ax : function
            Computes matrix-vector product Ax, where x is a vector of size n.
        matvec_ATx : function of x
            Computes matrix-vector product A^Tx, where x is a vector of size m.
        matmat_ATsA : function
            Computes matrix-matrix-matrix product A^T * Diag(s) * A,
    """

    def __init__(self, matvec_Ax, matvec_ATx, matmat_ATsA, b, regcoef):
        self.matvec_Ax = matvec_Ax
        self.matvec_ATx = matvec_ATx
        self.matmat_ATsA = matmat_ATsA
        self.b = b
        self.regcoef = regcoef
        self.m = b.size

    def func(self, x):
        z = np.asarray(self.matvec_Ax(x)).ravel()
        return np.mean(np.logaddexp(0, -self.b * z)) + 0.5 * self.regcoef * np.dot(x, x)

    def grad(self, x):
        z = np.asarray(self.matvec_Ax(x)).ravel()
        sigma = expit(-self.b * z)
        return np.asarray(self.matvec_ATx(-self.b * sigma)) / self.m + self.regcoef * x

    def hess(self, x):
        z = np.asarray(self.matvec_Ax(x)).ravel()
        sigma = expit(-self.b * z)
        s = sigma * (1.0 - sigma)
        return np.asarray(self.matmat_ATsA(s)) / self.m + self.regcoef * np.eye(x.size)


def grad_finite_diff(func, x, eps=1e-8):
    """
    Returns approximation of the gradient using finite differences:
        result_i := (f(x + eps * e_i) - f(x)) / eps,
        where e_i are coordinate vectors:
        e_i = (0, 0, ..., 0, 1, 0, ..., 0)
                          >> i <<
    """
    x = np.asarray(x, dtype='d')
    n = x.size
    result = np.zeros(n)
    fx = func(x)

    for i in range(n):
        x_eps = x.copy()
        x_eps[i] += eps
        result[i] = (func(x_eps) - fx) / eps

    return result


def hess_finite_diff(func, x, eps=1e-5):
    """
    Returns approximation of the Hessian using finite differences:
        result_{ij} := (f(x + eps * e_i + eps * e_j)
                               - f(x + eps * e_i) 
                               - f(x + eps * e_j)
                               + f(x)) / eps^2,
        where e_i are coordinate vectors:
        e_i = (0, 0, ..., 0, 1, 0, ..., 0)
                          >> i <<
    """
    x = np.asarray(x, dtype='d')
    n = x.size
    result = np.zeros((n, n))
    fx = func(x)
    eps2 = eps ** 2

    f_i = np.zeros(n)
    for i in range(n):
        x_eps = x.copy()
        x_eps[i] += eps
        f_i[i] = func(x_eps)

    for i in range(n):
        for j in range(n):
            x_eps_ij = x.copy()
            x_eps_ij[i] += eps
            x_eps_ij[j] += eps
            result[i][j] = (func(x_eps_ij) - f_i[i] - f_i[j] + fx) / eps2

    return result
