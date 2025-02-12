import numpy as np
import matplotlib.pyplot as plt


def rosenbrock(x):
    """Compute the Rosenbrock function."""
    return 100 * (x[1] - x[0] ** 2) ** 2 + (1 - x[0]) ** 2


def rosenbrock_grad(x):
    """Compute the gradient of the Rosenbrock function."""
    grad1 = -2 * (1 - x[0]) - 400 * (x[1] - x[0] ** 2) * x[0]
    grad2 = 200 * (x[1] - x[0] ** 2)
    return np.array([grad1, grad2])


def rosenbrock_hess(x):
    """Compute the Hessian of the Rosenbrock function."""
    h11 = 1200 * x[0] ** 2 - 400 * x[1] + 2
    h12 = -400 * x[0]
    h22 = 200
    return np.array([[h11, h12], [h12, h22]])


def GD(
    f,
    gradf,
    x,
    a0=1.0,
    c=1e-2,
    rho=0.1,
    tol=1e-6,
    max_iter=5000,
    max_iter_ls=10,
):
    """
    Gradient Descent with Armijo backtracking line search.

    Parameters:
      f: function to minimize.
      gradf: gradient of f.
      x: initial point.
      alpha0: initial step size.
      c: Armijo condition parameter.
      rho: contraction factor for step size.
      grad_stop: threshold for gradient norm.
      max_steps_outer: maximum iterations.
      max_steps_backtracking: maximum backtracking steps.

    Returns:
      x: approximate minimizer.
    """
    n = 0
    fi = f(x)
    Df = gradf(x)

    while n < max_iter:
        Nf = np.linalg.norm(Df)
        if Nf < tol:
            break
        a = a0
        xn = x - a * Df
        fn = f(xn)
        n_ls = 0

        # Backtracking until Armijo condition is met
        while fn > fi - c * a * Nf**2 and n_ls < max_iter_ls:
            a *= rho
            xn = x - a * Df
            fn = f(xn)
            n_ls += 1

        x = xn
        fi = fn
        Df = gradf(x)
        n += 1

    if n == max_iter:
        print("Algorithm stopped after reaching the maximum number of iterations.")
    else:
        print(f"Algorithm converged after {n} iterations.")
    return x


def Newton(
    f,
    gradf,
    hessf,
    x,
    c=1e-2,
    rho=0.1,
    tol=1e-6,
    max_iter=5000,
    max_iter_ls=20,
    epsilon_grad=1e-6,
):
    """
    Newton's method with Armijo backtracking line search.

    Parameters:
      f: function to minimize.
      gradf: gradient of f.
      hessf: Hessian of f.
      x: initial point.
      c: Armijo condition parameter.
      rho: contraction factor for step size.
      grad_stop: threshold for gradient norm.
      max_steps_outer: maximum iterations.
      max_steps_backtracking: maximum backtracking steps.
      epsilon_grad: tolerance for ensuring proper descent direction.

    Returns:
      x: approximate minimizer.
    """
    n = 0
    fi = f(x)
    Df = gradf(x)
    N_Df = np.linalg.norm(Df)

    while n < max_iter:
        if N_Df < tol:
            break

        try:  # Compute the Newton direction
            pk = -np.linalg.solve(hessf(x), Df)  # Newton direction
            qk = np.inner(pk, Df)  # Newton decrement
            if (
                qk >= -epsilon_grad * np.linalg.norm(pk) * N_Df
            ):  # Ensure descent direction
                pk = -Df
                qk = N_Df**2
        except np.linalg.LinAlgError:
            # If Hessian is singular, fallback to gradient descent
            pk = -Df
            qk = N_Df**2

        if qk <= 0:  # Ensure descent direction
            print("Switching to gradient descent due to non-positive initial descent.")
            pk = -Df
            qk = N_Df**2

        a = 1.0
        xn = x + a * pk
        fn = f(xn)
        n_ls = 0

        # Backtracking until Armijo condition is met
        while fn > fi - c * a * qk:
            if n_ls >= max_iter_ls:
                break
            a *= rho
            xn = x + a * pk
            fn = f(xn)
            n_ls += 1

        x = xn
        fi = fn
        Df = gradf(x)
        N_Df = np.linalg.norm(Df)
        n += 1

    if n == max_iter:
        print("Algorithm stopped after reaching the maximum number of iterations.")
    else:
        print(f"Algorithm converged after {n} iterations.")
    return x


# Initial point
x0 = np.array([-1.2, 1.0])

x_gd = GD(rosenbrock, rosenbrock_grad, x0)
x_nw = Newton(rosenbrock, rosenbrock_grad, rosenbrock_hess, x0)

# Plot the Rosenbrock function
x = np.linspace(-2, 2, 100)
y = np.linspace(-1, 3, 100)
X, Y = np.meshgrid(x, y)
Z = 100 * (Y - X**2) ** 2 + (1 - X) ** 2

plt.figure()
plt.contour(X, Y, Z, cmap="viridis", levels=100)
plt.plot(x_gd[0], x_gd[1], "ro", label="Gradient Descent")
plt.plot(x_nw[0], x_nw[1], "bx", label="Newton's Method", markersize=10)
plt.legend()
plt.show()
