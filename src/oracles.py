import numpy as np
import scipy
from scipy.special import expit


class BaseSmoothOracle(object):
    """Base class for smooth function."""
    def func(self, x):
        raise NotImplementedError('Func is not implemented.')

    def grad(self, x):
        raise NotImplementedError('Grad is not implemented.')

    def hess(self, x):
        raise NotImplementedError('Hess is not implemented.')

    def func_directional(self, x, d, alpha):
        return np.squeeze(self.func(x + alpha * d))

    def grad_directional(self, x, d, alpha):
        return np.squeeze(self.grad(x + alpha * d).dot(d))


class BaseProxOracle(object):
    """Base class for proximal h(x)-part in a composite function f(x) + h(x)."""
    def func(self, x):
        raise NotImplementedError('Func is not implemented.')

    def prox(self, x, alpha):
        raise NotImplementedError('Prox is not implemented.')


class BaseCompositeOracle(object):
    """phi(x) := f(x) + h(x), where f is a smooth part, h is a simple part."""
    def __init__(self, f, h):
        self._f = f
        self._h = h

    def func(self, x):
        return self._f.func(x) + self._h.func(x)

    def grad(self, x):
        return self._f.grad(x)

    def prox(self, x, alpha):
        return self._h.prox(x, alpha)


class BaseNonsmoothConvexOracle(object):
    """Base class for implementation of oracle for nonsmooth convex function."""
    def func(self, x):
        raise NotImplementedError('Func is not implemented.')

    def subgrad(self, x):
        raise NotImplementedError('Subgrad is not implemented.')


class QuadraticOracle(BaseSmoothOracle):
    """Oracle for quadratic function: func(x) = 1/2 x^TAx - b^Tx."""
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
    """Oracle for Himmelblau test function."""
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


class L1RegOracle(BaseProxOracle):
    """
    Oracle for L1-regularizer: h(x) = regcoef * ||x||_1.
    """
    def __init__(self, regcoef):
        self.regcoef = regcoef

    def func(self, x):
        return self.regcoef * np.linalg.norm(x, 1)

    def prox(self, x, alpha):
        """Soft-thresholding operator."""
        threshold = alpha * self.regcoef
        return np.sign(x) * np.maximum(np.abs(x) - threshold, 0.0)

    def lmo(self, grad, radius):
        """
        Linear Minimization Oracle for L1-ball.
        Returns y_k = argmin_{||y||_1 <= R} <grad, y>
        """
        i = np.argmax(np.abs(grad))
        y = np.zeros_like(grad)
        y[i] = -radius * np.sign(grad[i])
        return y


class BarrierL1Oracle(object):
    """
    Oracle for the barrier function in the barrier method:
    F_t(x, u) = t*(f(x) + lambda*sum(u)) - sum(ln(u_i - x_i)) - sum(ln(u_i + x_i))

    State vector z = [x, u] (concatenation).
    """
    def __init__(self, smooth_oracle, lambda_reg, t):
        self.smooth_oracle = smooth_oracle
        self.lambda_reg = lambda_reg
        self.t = t

    def _split(self, z):
        n = len(z) // 2
        return z[:n], z[n:]

    def func(self, z):
        x, u = self._split(z)
        d = u - x  # must be > 0
        s = u + x  # must be > 0
        return (self.t * (self.smooth_oracle.func(x) + self.lambda_reg * np.sum(u))
                - np.sum(np.log(d)) - np.sum(np.log(s)))

    def grad(self, z):
        x, u = self._split(z)
        d = u - x
        s = u + x
        gx = self.t * self.smooth_oracle.grad(x) + 1.0 / d - 1.0 / s
        gu = self.t * self.lambda_reg * np.ones_like(u) - 1.0 / d - 1.0 / s
        return np.concatenate([gx, gu])

    def hess(self, z):
        x, u = self._split(z)
        n = len(x)
        d = u - x
        s = u + x
        d2 = 1.0 / d ** 2
        s2 = 1.0 / s ** 2

        # Smooth part Hessian (n x n)
        Hf = np.asarray(self.smooth_oracle.hess(x))

        # Build 2n x 2n Hessian
        H = np.zeros((2 * n, 2 * n))

        # x-x block: t*Hf + diag(1/d^2 + 1/s^2)
        H[:n, :n] = self.t * Hf + np.diag(d2 + s2)

        # u-u block: diag(1/d^2 + 1/s^2)
        H[n:, n:] = np.diag(d2 + s2)

        # x-u block (and u-x): diag(-1/d^2 + 1/s^2)
        xu_diag = np.diag(-d2 + s2)
        H[:n, n:] = xu_diag
        H[n:, :n] = xu_diag

        return H


class RegressionSmoothOracle(BaseSmoothOracle):
    """Smooth oracle for Poisson regression loss with L2 regularization."""
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


class ClassificationSmoothOracle(BaseSmoothOracle):
    """Smooth oracle for logistic regression loss with L2 regularization."""
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


# Keep old names as aliases for backward compatibility
PoissonL2Oracle = RegressionSmoothOracle
LogisticL2Oracle = ClassificationSmoothOracle


class RegressionNonsmoothOracle(BaseNonsmoothConvexOracle):
    """regression_loss + regcoef * ||x||_1."""
    def __init__(self, matvec_Ax, matvec_ATx, matmat_ATsA, b, regcoef):
        self._smooth = RegressionSmoothOracle(matvec_Ax, matvec_ATx, matmat_ATsA, b, regcoef=0.0)
        self.regcoef = regcoef

    def func(self, x):
        return self._smooth.func(x) + self.regcoef * np.linalg.norm(x, 1)

    def subgrad(self, x):
        return self._smooth.grad(x) + self.regcoef * np.sign(x)


class ClassificationNonsmoothOracle(BaseNonsmoothConvexOracle):
    """classification_loss + regcoef * ||x||_1."""
    def __init__(self, matvec_Ax, matvec_ATx, matmat_ATsA, b, regcoef):
        self._smooth = ClassificationSmoothOracle(matvec_Ax, matvec_ATx, matmat_ATsA, b, regcoef=0.0)
        self.regcoef = regcoef

    def func(self, x):
        return self._smooth.func(x) + self.regcoef * np.linalg.norm(x, 1)

    def subgrad(self, x):
        return self._smooth.grad(x) + self.regcoef * np.sign(x)


class RegressionProxOracle(BaseCompositeOracle):
    """
    Oracle for regression_loss + regcoef * ||x||_1.
    f(x) = regression_loss (smooth, no L2), h(x) = regcoef * ||x||_1 (simple).
    """
    def __init__(self, matvec_Ax, matvec_ATx, matmat_ATsA, b, regcoef=1.0):
        f = RegressionSmoothOracle(matvec_Ax, matvec_ATx, matmat_ATsA, b, regcoef=0.0)
        h = L1RegOracle(regcoef)
        super().__init__(f, h)


class ClassificationProxOracle(BaseCompositeOracle):
    """
    Oracle for classification_loss + regcoef * ||x||_1.
    f(x) = classification_loss (smooth, no L2), h(x) = regcoef * ||x||_1 (simple).
    """
    def __init__(self, matvec_Ax, matvec_ATx, matmat_ATsA, b, regcoef=1.0):
        f = ClassificationSmoothOracle(matvec_Ax, matvec_ATx, matmat_ATsA, b, regcoef=0.0)
        h = L1RegOracle(regcoef)
        super().__init__(f, h)


def grad_finite_diff(func, x, eps=1e-8):
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
