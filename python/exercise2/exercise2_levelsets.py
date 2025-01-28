import dash
from dash import dcc, html
from dash.dependencies import Input, Output, State
import plotly.graph_objs as go
import numpy as np

app = dash.Dash(__name__)

app.layout = html.Div([
    dcc.Dropdown(
        id="function-select",
        options=[
            {"label": "Convex: x² + y²", "value": "convex"},
            {"label": "Quasi-convex: x/(1 + y²)", "value": "quasiconvex"},
        ],
        value="convex",
    ),
    dcc.Slider(
        id="alpha-slider",
        min=0,
        max=20,
        step=0.1,
        value=10,
        marks={i: str(i) for i in range(0, 26, 5)},
    ),
    dcc.Graph(id="3d-plot"),
    html.Div(id="click-message"),
    html.Div(id="stored-clicks", style={"display": "none"}),
])

@app.callback(
    Output("3d-plot", "figure"),
    [Input("function-select", "value"), Input("alpha-slider", "value")]
)
def update_3d(function, alpha):
    x = y = np.linspace(-5, 5, 50)
    X, Y = np.meshgrid(x, y)
    
    if function == "convex":
        Z = X**2 + Y**2
        zmax = 10
    else:
        Z = X / (1 + Y**2)
        zmax = 10

    # Create main surface plot
    surface = go.Surface(
        x=X,
        y=Y,
        z=Z,
        colorscale="Viridis",
        opacity=0.8,
        contours_z=dict(show=True, usecolormap=True, highlightcolor="limegreen")
    )

    # Create alpha plane
    alpha_plane = go.Surface(
        x=X,
        y=Y,
        z=np.full_like(Z, alpha),
        colorscale='reds',
        lighting=dict(ambient=0.5),
        showscale=False,
        opacity=0.5
    )  

    # Create sublevel set visualization
    sublevel = go.Surface(
        x=X,
        y=Y,
        z=np.where(Z <= alpha, Z, np.nan),
        colorscale="Viridis",
        showscale=False,
        opacity=0.9
    )

    fig = go.Figure(data=[surface, alpha_plane, sublevel])
    
    fig.update_layout(
        title=f"3D Visualization (α={alpha})",
        scene=dict(
            xaxis_title="X",
            yaxis_title="Y",
            zaxis_title="f(x,y)",
            zaxis=dict(range=[-5 if function == "quasiconvex" else 0, zmax]),
            camera=dict(
                eye=dict(x=1.5, y=1.5, z=0.8)
            )
        ),
        margin=dict(l=0, r=0, b=0, t=40)
    )
    
    return fig

@app.callback(
    Output("alpha-slider", "min"),
    Output("alpha-slider", "max"),
    Output("alpha-slider", "marks"),
    [Input("function-select", "value")]
)
def update_slider(function):
    if function == "quasiconvex":
        return -5, 5, {i: str(i) for i in range(-5, 6)}
    return 0, 25, {i: str(i) for i in range(0, 26, 5)}

@app.callback(
    Output("stored-clicks", "children"),
    [Input("3d-plot", "clickData")],
    [State("stored-clicks", "children")]
)
def store_clicks(clickData, stored_clicks):
    if not clickData:
        return stored_clicks
    point = clickData["points"][0]
    x, y = point["x"], point["y"]
    stored_clicks = eval(stored_clicks) if stored_clicks else []
    stored_clicks = stored_clicks[-1:] + [(x, y)]
    return str(stored_clicks)

@app.callback(
    Output("click-message", "children"),
    [Input("stored-clicks", "children"),
     Input("function-select", "value"),
     Input("alpha-slider", "value")]
)
def update_message(stored_clicks, function, alpha):
    if not stored_clicks or len(eval(stored_clicks)) < 2:
        return html.P("Click two points on the plot to check convexity.")
    
    p1, p2 = eval(stored_clicks)
    p1, p2 = np.array(p1), np.array(p2)
    t = np.linspace(0, 1, 100)
    pts = p1 + t[:, None] * (p2 - p1)
    
    if function == "convex":
        f = lambda x, y: x**2 + y**2
    else:
        f = lambda x, y: x / (1 + y**2)
    
    f_vals = f(pts[:, 0], pts[:, 1])
    in_sublevel = np.all(f_vals <= alpha)
    linear_interp = (1 - t)*f(*p1) + t*f(*p2)
    is_convex_func = np.all(f_vals <= linear_interp)
    
    messages = [
        f"Line segment is {'within' if in_sublevel else 'outside'} the sublevel set.",
        "Function is convex." if function == "convex" else 
        "Function is quasi-convex: sublevel sets are convex, but function may not be."
    ]
    
    return html.Div([html.P(m) for m in messages])

if __name__ == "__main__":
    app.run_server(debug=True)