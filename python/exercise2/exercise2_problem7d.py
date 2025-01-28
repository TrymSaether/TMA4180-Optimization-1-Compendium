import numpy as np
import plotly.graph_objects as go

# Create x values
x = np.linspace(-3, 3, 1000)

# Define the piecewise function
def f(x):
    return np.where(x > 0, x,
           np.where((-1 < x) & (x <= 0), 0,
           np.where((-2 < x) & (x <= -1), x + 1,
           -x - 3)))

# Calculate y values
y = f(x)

# Create the plot
fig = go.Figure()
fig.add_trace(go.Scatter(x=x, y=y, mode='lines', name='f(x)'))

# Update layout
fig.update_layout(
    title='Piecewise Function',
    xaxis_title='x',
    yaxis_title='f(x)',
    showlegend=True
)

# Add x and y axis lines
fig.add_hline(y=0, line_width=1, line_color='black')
fig.add_vline(x=0, line_width=1, line_color='black')

fig.show()