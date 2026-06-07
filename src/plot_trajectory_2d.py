import numpy as np
import matplotlib.pyplot as plt


def plot_levels(func, xrange=None, yrange=None, levels=None):
    """
    Plotting the contour lines of the function.

    Example:
    --------
    >> oracle = oracles.QuadraticOracle(np.array([[1.0, 2.0], [2.0, 5.0]]), np.zeros(2))
    >> plot_levels(oracle.func)
    """
    if xrange is None:
        xrange = [-6, 6]
    if yrange is None:
        yrange = [-5, 5]

    x = np.linspace(*xrange, 200)
    y = np.linspace(*yrange, 200)
    X, Y = np.meshgrid(x, y)

    grid = np.c_[X.ravel(), Y.ravel()]
    Z = np.array([func(p) for p in grid]).reshape(X.shape)

    plt.contour(X, Y, Z, levels=levels, cmap='viridis', linewidths=0.8)

    plt.gca().set_facecolor('white')
    plt.grid(alpha=0.3)


def plot_trajectory(func, history, fit_axis=False, label=None, color=None, linestyle=None):
    """
    Plotting the trajectory of a method. 
    Use after plot_levels(...).

    Example:
    --------
    >> oracle = oracles.QuadraticOracle(np.array([[1.0, 2.0], [2.0, 5.0]]), np.zeros(2))
    >> [x_star, msg, history] = optimization.gradient_descent(oracle, np.array([3.0, 1.5], trace=True)
    >> plot_levels(oracle.func)
    >> plot_trajectory(oracle.func, history['x'])
    """
    x_values, y_values = zip(*history)

    plt.plot(x_values, y_values,
             markersize=4,
             linewidth=1,
             color=color,
             linestyle=linestyle,
             label=label)

    plt.quiver(
        x_values[:-1], y_values[:-1],
        np.diff(x_values), np.diff(y_values),
        angles='xy', scale_units='xy', scale=1,
        color=color, alpha=0.5
    )

    plt.scatter(x_values[0], y_values[0], color=color, marker='s', s=60)
    plt.scatter(x_values[-1], y_values[-1], color=color, marker='*', s=100)
