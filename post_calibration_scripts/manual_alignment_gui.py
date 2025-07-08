# This GUI is a work in process
# but can be used for manual alignment of the ant images
# the output of this would typically serve as a start point
# for fine alignment

import dash
from dash import dcc, html
from dash.dependencies import Input, Output, State
import plotly.graph_objects as go
import numpy as np
import trimesh

from hsflfm.ant_model import mesh_filename
from hsflfm.config import home_directory
from hsflfm.processing import Aligner
from hsflfm.util import rot_trans_from_matrix, matrix_from_rot_trans
import math
import scipy
from scipy.spatial import KDTree

import sys

# Initialize the Dash app
app = dash.Dash(__name__)

# load mesh
mesh = trimesh.load(home_directory + "/" + mesh_filename)
mesh_x = mesh.vertices[:, 0]
mesh_y = mesh.vertices[:, 1]
mesh_z = mesh.vertices[:, 2]

# Initial dynamic points data
points_x, points_y, points_z = np.random.rand(3, 40)  # Example points data
mesh_faces = mesh.faces

# set up the tree
sample_vertices, _ = trimesh.sample.sample_surface(
    mesh,
    len(mesh.vertices) * 100,
)
tree = KDTree(sample_vertices)

if len(sys.argv) > 1:
    specimen_number = sys.argv[1]
    print(f"[INFO] Loading Speciment: {specimen_number}")
else: 
    specimen_number = "20240506_OB_6" #Manually specify 

aligner = Aligner(specimen_number)

#A_base, s_base = aligner.run_strike1_alignment()  
A_base, s_base = aligner.run_base_alignment()
mp1 = aligner.move_points_to_mesh(A_base, s_base, aligner.point_camera_locations)

# Create the figure with a static mesh and dynamic points
fig = go.Figure(
    data=[
        # Static Mesh
        go.Mesh3d(
            x=mesh_x,
            y=mesh_y,
            z=mesh_z,
            i=mesh_faces[:, 0],
            j=mesh_faces[:, 1],
            k=mesh_faces[:, 2],
            opacity=0.5,
            color="lightblue",
            name="Mesh",
        ),
        # Base Points
        go.Scatter3d(
            x=mp1[:, 0],
            y=mp1[:, 1],
            z=mp1[:, 2],
            mode="markers",
            marker=dict(size=5, color="blue"),
            name="Base Points",
        ),
        # Dynamic Points
        go.Scatter3d(
            x=mp1[:, 0],
            y=mp1[:, 1],
            z=mp1[:, 2],
            mode="markers",
            marker=dict(size=5, color="red"),
            name="Dynamic Points",
        ),
    ]
)

# Layout with sliders, input boxes, and plot
app.layout = html.Div(
    [
        # Container for sliders and inputs, two rows of sliders
        html.Div(
            [
                # First Row of Sliders
                html.Div(
                    [
                        html.Label("X"),
                        dcc.Slider(
                            id="x-slider",
                            min=-0.5,
                            max=0.5,
                            step=0.001,
                            value=0,
                            marks={-0.5: "-0.5", 0: "0", 0.5: "0.5"},
                        ),
                        dcc.Input(
                            id="x-input",
                            type="number",
                            value=0,
                            style={"width": "70px"},
                            debounce=True,
                        ),
                    ],
                    style={"flex": "1", "padding": "10px"},
                ),
                html.Div(
                    [
                        html.Label("Y"),
                        dcc.Slider(
                            id="y-slider",
                            min=-1,
                            max=1,
                            step=0.001,
                            value=0,
                            marks={-1: "-1", 0: "0", 1: "1"},
                        ),
                        dcc.Input(
                            id="y-input",
                            type="number",
                            value=0,
                            style={"width": "70px"},
                            debounce=True,
                        ),
                    ],
                    style={"flex": "1", "padding": "10px"},
                ),
                html.Div(
                    [
                        html.Label("Z"),
                        dcc.Slider(
                            id="z-slider",
                            min=-1,
                            max=1,
                            step=0.001,
                            value=0,
                            marks={-1: "-1", 0: "0", 1: "1"},
                        ),
                        dcc.Input(
                            id="z-input",
                            type="number",
                            value=0,
                            style={"width": "70px"},
                            debounce=True,
                        ),
                    ],
                    style={"flex": "1", "padding": "10px"},
                ),
            ],
            style={
                "display": "flex",
                "justifyContent": "space-between",
                "padding": "10px",
            },
        ),
        html.Div(
            [
                # Second Row of Sliders
                html.Div(
                    [
                        html.Label("Roll"),
                        dcc.Slider(
                            id="roll-slider",
                            min=-10,
                            max=10,
                            step=0.1,
                            value=0,
                            marks={-10: "-10", 0: "0", 10: "10"},
                        ),
                        dcc.Input(
                            id="roll-input",
                            type="number",
                            value=0,
                            style={"width": "70px"},
                            debounce=True,
                        ),
                    ],
                    style={"flex": "1", "padding": "10px"},
                ),
                html.Div(
                    [
                        html.Label("Pitch"),
                        dcc.Slider(
                            id="pitch-slider",
                            min=-10,
                            max=10,
                            step=0.1,
                            value=0,
                            marks={-10: "-10", 0: "0", 10: "10"},
                        ),
                        dcc.Input(
                            id="pitch-input",
                            type="number",
                            value=0,
                            style={"width": "70px"},
                            debounce=True,
                        ),
                    ],
                    style={"flex": "1", "padding": "10px"},
                ),
                html.Div(
                    [
                        html.Label("Yaw"),
                        dcc.Slider(
                            id="yaw-slider",
                            min=-10,
                            max=10,
                            step=0.1,
                            value=0,
                            marks={-10: "-10", 0: "0", 10: "10"},
                        ),
                        dcc.Input(
                            id="yaw-input",
                            type="number",
                            value=0,
                            style={"width": "70px"},
                            debounce=True,
                        ),
                    ],
                    style={"flex": "1", "padding": "10px"},
                ),
            ],
            style={
                "display": "flex",
                "justifyContent": "space-between",
                "padding": "10px",
            },
        ),
        # Display Loss Value
        html.Div(
            [
                html.Label("Loss:"),
                dcc.Input(
                    id="loss-input",
                    type="number",
                    value=0,
                    style={"width": "70px", "fontWeight": "bold"},
                    disabled=True,
                ),
            ],
            style={"padding": "10px"},
        ),
        # Plot
        dcc.Graph(id="3d-plot", figure=fig),
    ]
)


# Callback for sliders to update plot dynamically
@app.callback(
    [Output("3d-plot", "figure"), Output('loss-input', 'value')],
    [
        Input("x-slider", "value"),
        Input("y-slider", "value"),
        Input("z-slider", "value"),
        Input("roll-slider", "value"),
        Input("pitch-slider", "value"),
        Input("yaw-slider", "value"),
    ],
    [State("3d-plot", "relayoutData")],
)
def update_plot_from_slider(dx, dy, dz, droll, dpitch, dyaw, relayout_data):
    droll = droll * math.pi / 180
    dpitch = dpitch * math.pi / 180
    dyaw = dyaw * math.pi / 180
    x, y, z, roll, pitch, yaw = rot_trans_from_matrix(np.linalg.inv(A_base))
    A_new = matrix_from_rot_trans(
        x + dx, y + dy, z + dz, roll + droll, pitch + dpitch, yaw + dyaw
    )

    mp2 = aligner.move_points_to_mesh(
        np.linalg.inv(A_new), s_base, aligner.point_camera_locations
    )

    fig.update_traces(
        selector=dict(name="Dynamic Points"), x=mp2[:, 0], y=mp2[:, 1], z=mp2[:, 2]
    )

    # Preserve camera view if relayout data has camera settings
    if relayout_data and "scene.camera" in relayout_data:
        fig.update_layout(scene_camera=relayout_data["scene.camera"])


    # get loss
    distances, _ = tree.query(mp2) 
    huber_delta = 4000
    base_loss = scipy.special.huber(huber_delta, distances)
    base_loss[2] = 0 
    base_loss = np.mean(base_loss)
    print(base_loss / 1e6)
    #print(x, y, z, roll, pitch, yaw)
    #print(dx, dy, dz, droll, dpitch, dyaw)
    print(x + dx, y + dy, z + dz, roll + droll, pitch + dpitch, yaw + dyaw)
    print()
    return fig, base_loss / 1e3


# Callback to update sliders based on input values after pressing Enter
@app.callback(
    [
        Output("x-slider", "value"),
        Output("y-slider", "value"),
        Output("z-slider", "value"),
        Output("roll-slider", "value"),
        Output("pitch-slider", "value"),
        Output("yaw-slider", "value"),
    ],
    [
        Input("x-input", "n_submit"),
        Input("y-input", "n_submit"),
        Input("z-input", "n_submit"),
        Input("roll-input", "n_submit"),
        Input("pitch-input", "n_submit"),
        Input("yaw-input", "n_submit"),
    ],
    [
        Input("x-input", "value"),
        Input("y-input", "value"),
        Input("z-input", "value"),
        Input("roll-input", "value"),
        Input("pitch-input", "value"),
        Input("yaw-input", "value"),
    ],
)
def update_sliders_from_input(
    x_submit, y_submit, z_submit, roll_submit, pitch_submit, yaw_submit, x_value, y_value, z_value, roll_value, pitch_value, yaw_value
):
    return x_value, y_value, z_value, roll_value, pitch_value, yaw_value


if __name__ == "__main__":
    app.run(debug=True)
