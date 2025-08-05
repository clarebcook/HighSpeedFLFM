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
from pathlib import Path
import json

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

# Initialize results dictionary
alignment_results = {}

# set up the tree
sample_vertices, _ = trimesh.sample.sample_surface(
    mesh,
    len(mesh.vertices) * 100,
)
tree = KDTree(sample_vertices)

assert len(sys.argv) > 1
specimen_number = sys.argv[1]

aligner = Aligner(specimen_number)

A_base, s_base = aligner.run_strike1_alignment()
# A_base, s_base = aligner.run_base_alignment()
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

empty_refined_fig = go.Figure(
    data=[
        go.Mesh3d(
            x=mesh_x,
            y=mesh_y,
            z=mesh_z,
            i=mesh_faces[:, 0],
            j=mesh_faces[:, 1],
            k=mesh_faces[:, 2],
            opacity=0.5,
            color="lightgray",
            name="Ant Mesh",
        )
    ],
    layout=go.Layout(
        title="Refined Alignment Output",
        scene=dict(
            xaxis_title="X",
            yaxis_title="Y",
            zaxis_title="Z",
            aspectmode="data",
        ),
    )
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
        html.Button("Done", id="done-button"),
        html.Button("Refine Alignment", id="refine-button", style={"marginLeft": "10px"}),
        html.Div(id="status-div"),
        dcc.Graph(id="refined-plot", figure=empty_refined_fig, style={"height": "500px"}),

    ]
)


# Callback for sliders to update plot dynamically
@app.callback(
    [Output("3d-plot", "figure"), Output("loss-input", "value")],
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

    alignment_results.update(
        {
            "Specimen-Number": specimen_number,
            "x": x + dx,
            "y": y + dy,
            "z": z + dz,
            "roll": roll + droll,
            "pitch": pitch + dpitch,
            "yaw": yaw + dyaw,
            "base_loss": base_loss,
        }
    )

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
    x_submit,
    y_submit,
    z_submit,
    roll_submit,
    pitch_submit,
    yaw_submit,
    x_value,
    y_value,
    z_value,
    roll_value,
    pitch_value,
    yaw_value,
):
    return x_value, y_value, z_value, roll_value, pitch_value, yaw_value


# Done Button Functionality
@app.callback(
    Output("status-div", "children"),
    Input("done-button", "n_clicks"),
    prevent_initial_call=True,
)
def on_done_click(n_clicks):
    output_path = Path(__file__).parent.resolve() / "alignment_output.json"

    result = alignment_results.copy()
    result["Specimen-Number"] = specimen_number
    result["base_loss"] = result["base_loss"] / 1e6

    with open(output_path, "w") as f:
        json.dump(result, f, indent=2)

    return f"Alignment values saved for {result["Specimen-Number"]}. You may now close the window."


# Refine Alignment button functionality
@app.callback(
    Output("refined-plot", "figure"),
    Input("refine-button", "n_clicks"),
    [
        State("x-slider", "value"),
        State("y-slider", "value"),
        State("z-slider", "value"),
        State("roll-slider", "value"),
        State("pitch-slider", "value"),
        State("yaw-slider", "value"),
    ],
    prevent_initial_call=True,
)
def refine_alignment_display_only(n_clicks, dx, dy, dz, droll, dpitch, dyaw):
    # Convert degrees to radians
    droll = droll * math.pi / 180
    dpitch = dpitch * math.pi / 180
    dyaw = dyaw * math.pi / 180

    # Build matrix from user-provided deltas
    x, y, z, roll, pitch, yaw = rot_trans_from_matrix(A_base)
    A_user = matrix_from_rot_trans(
        x + dx, y + dy, z + dz,
        roll + droll, pitch + dpitch, yaw + dyaw
    )

    # Run fine alignment
    A_refined, _ = aligner.refine_matrix(
        A_cam_ant_init=A_user,
        ant_scale_init=s_base,
        camera_points=aligner.point_camera_locations,
        change_scale=False,
    )

    # Transform points using refined matrix
    mp_refined = aligner.move_points_to_mesh(
        A_refined,
        s_base,
        aligner.point_camera_locations,
    )

    # Create new 3D figure with mesh + refined points
    fig = go.Figure(
        data=[
            # Ant CT mesh
            go.Mesh3d(
                x=mesh_x,
                y=mesh_y,
                z=mesh_z,
                i=mesh_faces[:, 0],
                j=mesh_faces[:, 1],
                k=mesh_faces[:, 2],
                opacity=0.5,
                color="lightblue",
                name="Ant Mesh",
            ),
            # Refined points
            go.Scatter3d(
                x=mp_refined[:, 0],
                y=mp_refined[:, 1],
                z=mp_refined[:, 2],
                mode="markers",
                marker=dict(size=5, color="green"),
                name="Refined Points",
            ),
        ],
        layout=go.Layout(
            title="Refined Alignment Output",
            scene=dict(
                xaxis_title="X",
                yaxis_title="Y",
                zaxis_title="Z",
                aspectmode="data",  # ensure mesh proportions aren't distorted
            ),
        ),
    )

    return fig



if __name__ == "__main__":
    app.run(debug=False)
