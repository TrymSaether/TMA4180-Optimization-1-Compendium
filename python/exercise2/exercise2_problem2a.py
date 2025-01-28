import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots

# Define a simple convex function: f(x, y) = x^2 + y^2
def f(x, y):
    return x**2 + y**2

# Generate grid points for the function
a0, a1 = -1, 1
b0, b1 = -1, 1

x = np.linspace(a0, a1, 100)
y = np.linspace(b0, b1, 100)
X, Y = np.meshgrid(x, y)
Z = f(X, Y)

# Define two points in the set of minimizers (global minimum is at (0, 0))
minimizer1 = np.array([0, 0])
minimizer2 = np.array([0.5, 0.5])

# Compute the convex combination of the two points
lambdas = np.linspace(0, 1, 101)
convex_combinations = [l * minimizer1 + (1 - l) * minimizer2 for l in lambdas]

# Extract x and y values of the convex combinations
convex_x = [p[0] for p in convex_combinations]
convex_y = [p[1] for p in convex_combinations]

# Create the figure
fig = go.Figure()

# Add surface plot
fig.add_trace(
    go.Surface(
        x=X, y=Y, z=Z,
        colorscale='viridis',
        opacity=0.8,
    )
)

# Add minimizers as scatter points
z_values = [f(x, y) for x, y in convex_combinations]
fig.add_trace(
    go.Scatter3d(
        x=[minimizer1[0]], y=[minimizer1[1]], z=[f(minimizer1[0], minimizer1[1])],
        mode='markers',
        marker=dict(size=8, color='red'),
        name='x₁'
    )
)
fig.add_trace(
    go.Scatter3d(
        x=[minimizer2[0]], y=[minimizer2[1]], z=[f(minimizer2[0], minimizer2[1])],
        mode='markers',
        marker=dict(size=8, color='blue'),
        name='x₂'
    )
)

# Add convex combination line
fig.add_trace(
    go.Scatter3d(
        x=convex_x, y=convex_y, z=z_values,
        mode='lines',
        line=dict(color='green', width=4),
        name='λx₁ + (1-λ)x₂'
    )
)

# Add lambda points and annotations
for i, l in enumerate(lambdas):
    if i % 10 == 0:
        x, y = convex_combinations[i]
        z = f(x, y)
        fig.add_trace(
            go.Scatter3d(
                x=[x], y=[y], z=[z],
                mode='markers+text',
                marker=dict(size=4, color='red'),
                text=[f'λ={l:.1f}', f'f(x, y)={z:.2f}'],
                textposition='top center',
                textfont=dict(size=10, color='black'),
                showlegend=False,    
            )
        )

# Update layout
fig.update_layout(
    title='Convex combination of two points - 3D visualization',
    scene=dict(
        xaxis_title='x',
        yaxis_title='y',
        zaxis_title='f(x,y)',
        camera=dict(
            eye=dict(x=1.5, y=1.5, z=1.5),
        )
    ),
    width=800,
    height=800,
)

# Save as HTML (for interactive visualization) and show
fig.write_html("figures/exercise2_problem2a_3d.html")
fig.show()
