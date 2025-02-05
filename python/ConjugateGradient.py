import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import scienceplots
plt.style.use('science')

def rosenbrock(x):
    return 100.0 * (x[1] - x[0]**2)**2 + (1 - x[0])**2

def rosenbrock_grad(x):
    grad = np.zeros_like(x)
    grad[0] = -400.0 * x[0]*(x[1] - x[0]**2) - 2.0*(1 - x[0])
    grad[1] = 200.0 * (x[1] - x[0]**2)
    return grad

def strong_wolfe_line_search(f, grad_f, xk, pk, c1=1e-4, c2=0.4,
                             alpha0=1.0, max_iter=20):
    """
    A bracketing + zoom line search that finds alpha satisfying the
    strong Wolfe conditions:
      f(xk + alpha pk) <= f(xk) + c1 alpha gk^T pk       (sufficient decrease)
      |grad_f(xk + alpha pk)^T pk| <= c2 |gk^T pk|       (curvature condition)
    """
    # Initial checks
    gk = grad_f(xk)
    phi0 = f(xk)            # f(xk)
    dphi0 = np.dot(gk, pk)  # gk^T pk (directional derivative at alpha=0)

    def phi(alpha):
        return f(xk + alpha*pk)

    def dphi(alpha):
        return np.dot(grad_f(xk + alpha*pk), pk)

    # If dphi0 >= 0, the direction pk might not be descent.
    if dphi0 > 0:
        print("Warning: pk is not a descent direction!")
    
    # Bracketing phase
    alpha_prev = 0.0
    alpha_curr = alpha0
    phi_prev = phi0
    for i in range(max_iter):
        phi_curr = phi(alpha_curr)
        # Check Armijo
        if (phi_curr > phi0 + c1*alpha_curr*dphi0) or \
           (i > 0 and phi_curr >= phi_prev):
            # We have an interval [alpha_prev, alpha_curr] where the step could lie
            alpha_lo = alpha_prev
            alpha_hi = alpha_curr
            break
        
        # Check curvature (strong Wolfe part 2: abs(dphi(alpha)) <= c2 * abs(dphi0))
        dphi_curr = dphi(alpha_curr)
        if abs(dphi_curr) <= c2*abs(dphi0):
            # Satisfies strong Wolfe => we're done
            return alpha_curr
        if dphi_curr >= 0:
            # We found a sign change in derivative => bracket is [alpha_curr, alpha_prev]
            alpha_lo = alpha_curr
            alpha_hi = alpha_prev
            break
        
        # Otherwise, increase alpha
        alpha_prev = alpha_curr
        phi_prev = phi_curr
        alpha_curr *= 2.0
    else:
        # If we never broke, just return alpha_curr
        return alpha_curr

    # "Zoom" phase: refine within [alpha_lo, alpha_hi]
    for i in range(max_iter):
        alpha_j = 0.5*(alpha_lo + alpha_hi)  # bisection (could do cubic interp)
        phi_j = phi(alpha_j)

        if (phi_j > phi0 + c1*alpha_j*dphi0) or (phi_j >= phi(alpha_lo)): 
            alpha_hi = alpha_j
        else:
            dphi_j = dphi(alpha_j) 
            if abs(dphi_j) <= c2*abs(dphi0): 
                return alpha_j
            if dphi_j*(alpha_hi - alpha_lo) >= 0:
                alpha_hi = alpha_lo
            alpha_lo = alpha_j

    # If we exit loop, return best found so far
    return 0.5*(alpha_lo + alpha_hi)


def fletcher_reeves(f, grad_f, x0, c1=1e-4, c2=0.4, tol=1e-6, max_iters=2000):
    """
    Non-linear conjugate gradient (Fletcher–Reeves) for minimizing f in R^2.
    Uses strong Wolfe line search.
    
    :param f: Function to minimize
    :param grad_f: Gradient of the function
    :param x0: Initial guess (numpy array of shape (2,))
    :param c1, c2: Strong Wolfe constants (0 < c1 < c2 < 1/2)
    :param tol: Convergence tolerance (on gradient norm)
    :param max_iters: Maximum number of iterations
    :return: (x_opt, f_values, xs) where
             x_opt is the approximate minimizer,
             f_values is a list of function values per iteration,
             xs is a list of iterates for analysis or plotting.
    """
    xk = x0.copy()
    gk = grad_f(xk)
    pk = -gk
    fk = f(xk)

    f_values = [fk]
    xs = [xk.copy()]

    for k in range(max_iters):
        # Check convergence
        if np.linalg.norm(gk) < tol:
            break

        # Line search for alpha_k
        alpha_k = strong_wolfe_line_search(
            f, grad_f, xk, pk, c1=c1, c2=c2, alpha0=1.0, max_iter=20
        )

        # Update x
        xk_new = xk + alpha_k * pk
        gk_new = grad_f(xk_new)
        fk_new = f(xk_new)

        # Fletcher-Reeves beta
        beta_k = np.dot(gk_new, gk_new) / np.dot(gk, gk)

        # Update direction
        pk_new = -gk_new + beta_k * pk

        # Prepare next iteration
        xk, gk, pk = xk_new, gk_new, pk_new
        fk = fk_new

        f_values.append(fk)
        xs.append(xk.copy())

    return xk, f_values, xs

# Test on Rosenbrock
x0 = np.array([-1.2, 1.0])  # Classic bad initial guess for Rosenbrock
x_opt, f_vals, xs = fletcher_reeves(
    rosenbrock, rosenbrock_grad, x0, c1=1e-4, c2=0.4, tol=1e-8, max_iters=1000
)
print("Optimal x:", x_opt.round(3))
print("Optimal f(x):", rosenbrock(x_opt).round(3))
print("Number of iterations:", len(f_vals))

# Animate the optimization path
fig, ax = plt.subplots()
ax.set_xlim(-2, 2)
ax.set_ylim(-1, 3)
ax.set_xlabel("x")
ax.set_ylabel("y")
ax.set_title("Fletcher-Reeves optimization of Rosenbrock function")

# Plot Rosenbrock contours
x = np.linspace(-2, 2, 100)
y = np.linspace(-1, 3, 100)
X, Y = np.meshgrid(x, y)
Z = rosenbrock(np.array([X, Y]))
contours = ax.contourf(X, Y, Z, cmap="viridis", levels=50, alpha=0.8)
ax.plot(1, 1, 'bx')  # Optimal point

# Plot optimization path
line, = ax.plot([], [], 'r-o', lw=2, label="Optimization path")

def init():
    line.set_data([], [])
    return line,

def animate(i):
    x_coords = [x[0] for x in xs[:i+1]]
    y_coords = [x[1] for x in xs[:i+1]]
    line.set_data(x_coords, y_coords)
    return line,

ani = animation.FuncAnimation(fig, animate, frames=len(xs), init_func=init, blit=True)
plt.show()



