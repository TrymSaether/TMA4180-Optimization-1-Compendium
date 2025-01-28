import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from matplotlib.patches import Polygon
from IPython.display import HTML

# A "cooler" test function (Himmelblau's Function)
def f(x, y):
    return (x**2 + y - 11)**2 + (x + y**2 - 7)**2

# Create mesh grid for contour plot

x = np.linspace(-5, 5, 100)
y = np.linspace(-5, 5, 100)
X, Y = np.meshgrid(x, y)
Z = f(X, Y)

class NelderMead:
    def __init__(self, f, x0, step=0.5):
        self.f = f
        self.simplex = self._initialize_simplex(x0, step)
        self.centroid = None
        self.reflected = None
        self.expanded = None
        self.contracted = None
        
    def _initialize_simplex(self, x0, step):
        p1 = np.array(x0)
        p2 = p1 + np.array([step, 0])
        p3 = p1 + np.array([0, step])
        return np.array([p1, p2, p3])
    
    def step(self):
        fvals = [self.f(p[0], p[1]) for p in self.simplex]
        idx = np.argsort(fvals)
        self.simplex = self.simplex[idx]
        
        self.centroid = np.mean(self.simplex[:-1], axis=0)
        
        alpha = 1.0
        self.reflected = self.centroid + alpha * (self.centroid - self.simplex[-1])
        f_reflected = self.f(*self.reflected)
        
        if self.f(*self.simplex[0]) <= f_reflected < self.f(*self.simplex[-2]):
            self.simplex[-1] = self.reflected
            return "reflection"
        
        if f_reflected < self.f(*self.simplex[0]):
            gamma = 2.0
            self.expanded = self.centroid + gamma * (self.reflected - self.centroid)
            if self.f(*self.expanded) < f_reflected:
                self.simplex[-1] = self.expanded
                return "expansion"
            else:
                self.simplex[-1] = self.reflected
                return "reflection"
        
        beta = 0.5
        self.contracted = self.centroid + beta * (self.simplex[-1] - self.centroid)
        if self.f(*self.contracted) < self.f(*self.simplex[-1]):
            self.simplex[-1] = self.contracted
            return "contraction"
        
        sigma = 0.5
        v1 = self.simplex[0]
        self.simplex[1] = v1 + sigma * (self.simplex[1] - v1)
        self.simplex[2] = v1 + sigma * (self.simplex[2] - v1)
        return "shrink"

fig, ax = plt.subplots()
ax.contour(X, Y, Z, levels=50)
nm = NelderMead(f, [-3, -3])

simplex_plot = ax.fill(nm.simplex[:,0], nm.simplex[:,1], 'b', alpha=0.3)[0]
centroid_plot, = ax.plot([], [], 'ko')
reflected_plot, = ax.plot([], [], 'ro')
expanded_plot, = ax.plot([], [], 'go')
contracted_plot, = ax.plot([], [], 'yo')
title = ax.set_title('')

def init():
    ax.set_xlim(-5, 5)
    ax.set_ylim(-5, 5)
    return simplex_plot, centroid_plot, reflected_plot, expanded_plot, contracted_plot, title

def update(frame):
    action = nm.step()
    simplex_plot.set_xy(nm.simplex)
    
    if nm.centroid is not None:
        centroid_plot.set_data([nm.centroid[0]], [nm.centroid[1]])
    if nm.reflected is not None:
        reflected_plot.set_data([nm.reflected[0]], [nm.reflected[1]])
    if nm.expanded is not None:
        expanded_plot.set_data([nm.expanded[0]], [nm.expanded[1]])
    if nm.contracted is not None:
        contracted_plot.set_data([nm.contracted[0]], [nm.contracted[1]])
    
    title.set_text(f'Step {frame+1}: {action}')
    return simplex_plot, centroid_plot, reflected_plot, expanded_plot, contracted_plot, title

anim = FuncAnimation(fig, update, frames=50, interval=500, init_func=init, blit=False)
plt.show()

