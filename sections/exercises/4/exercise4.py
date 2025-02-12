import numpy as np
from scipy.linalg import hilbert

def conjugate_gradient(A, b, x0, tol=1e-6, max_iter=10000):
    r = b - A @ x0
    if np.linalg.norm(r) < tol:
        return x0, 0
    
    p = r
    x = x0
    k = 0
    
    while k < max_iter:
        Ap = A @ p
        alpha = (r @ r) / (p @ Ap)
        x = x + alpha * p
        r_new = r - alpha * Ap
        
        if np.linalg.norm(r_new) < tol:
            return x, k
            
        beta = (r_new @ r_new) / (r @ r)
        p = r_new + beta * p
        r = r_new
        k += 1
    
    return x, k    


N = [5,8,12,20]
for n in N:
    A = hilbert(n)
    b = np.ones(n)
    x0 = np.zeros(n)
    x, k = conjugate_gradient(A, b, x0)
    print(f'n = {n}, iter = {k}')
    

def rosenbrock(x1, x2):
    return 100*(x2 - x1**2)**2 + (1 - x1)**2

def f_grad(x1, x2): 
    return np.array([-400*x1*(x2 - x1**2) - 2*(1 - x1), 200*(x2 - x1**2)])

def fletcher_reeves(f, f_grad, x0, tol=1e-6, max_iter=10000):
    x = x0
    k = 0
    grad = f_grad(x[0], x[1])
    d = -grad
    while k < max_iter:
        alpha = 0.01
        x_new = x + alpha * d
        grad_new = f_grad(x_new[0], x_new[1])
        beta = (grad_new @ grad_new) / (grad @ grad)
        d = -grad_new + beta * d
        x = x_new
        grad = grad_new
        if np.linalg.norm(grad) < tol:
            return x, k
        k += 1
    return x, k

