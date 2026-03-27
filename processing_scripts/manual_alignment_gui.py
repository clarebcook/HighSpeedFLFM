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
    x, y, z, roll, pitch, yaw = rot_trans_from_matrix(A_base)
    A_new = matrix_from_rot_trans(
        x + dx, y + dy, z + dz, roll + droll, pitch + dpitch, yaw + dyaw
    )
    mp2 = aligner.move_points_to_mesh(
        A_new, s_base, aligner.point_camera_locations
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


def compute_slider_values_from_alignment(x, y, z, roll, pitch, yaw, A_base):
    x0, y0, z0, roll0, pitch0, yaw0 = rot_trans_from_matrix(A_base)

    dx = x - x0
    dy = y - y0
    dz = z - z0
    droll = (roll - roll0) * 180 / math.pi
    dpitch = (pitch - pitch0) * 180 / math.pi
    dyaw = (yaw - yaw0) * 180 / math.pi

    return (
        dx, dy, dz,
        droll, dpitch, dyaw,
        dx, dy, dz,
        droll, dpitch, dyaw,
    )


# Callback to update sliders based on input values after pressing Enter
@app.callback(
    [
        Output("x-slider", "value"),
        Output("y-slider", "value"),
        Output("z-slider", "value"),
        Output("roll-slider", "value"),
        Output("pitch-slider", "value"),
        Output("yaw-slider", "value"),
        Output("x-input", "value"),
        Output("y-input", "value"),
        Output("z-input", "value"),
        Output("roll-input", "value"),
        Output("pitch-input", "value"),
        Output("yaw-input", "value"),
    ],
    [
        Input("refine-button", "n_clicks"),
        Input("x-input", "n_submit"),
        Input("y-input", "n_submit"),
        Input("z-input", "n_submit"),
        Input("roll-input", "n_submit"),
        Input("pitch-input", "n_submit"),
        Input("yaw-input", "n_submit"),
        Input("x-slider", "value"),
        Input("y-slider", "value"),
        Input("z-slider", "value"),
        Input("roll-slider", "value"),
        Input("pitch-slider", "value"),
        Input("yaw-slider", "value"),
    ],
    [
        State("x-input", "value"),
        State("y-input", "value"),
        State("z-input", "value"),
        State("roll-input", "value"),
        State("pitch-input", "value"),
        State("yaw-input", "value"),
        State("x-slider", "value"),
        State("y-slider", "value"),
        State("z-slider", "value"),
        State("roll-slider", "value"),
        State("pitch-slider", "value"),
        State("yaw-slider", "value"),
    ],
)

def sync_slider_input(
    refine_click,
    x_submit, y_submit, z_submit,
    roll_submit, pitch_submit, yaw_submit,
    x_slider, y_slider, z_slider,
    roll_slider, pitch_slider, yaw_slider,
    x_input, y_input, z_input,
    roll_input, pitch_input, yaw_input,
    x_slider_prev, y_slider_prev, z_slider_prev,
    roll_slider_prev, pitch_slider_prev, yaw_slider_prev,
):
    ctx = dash.callback_context

    if not ctx.triggered:
        raise dash.exceptions.PreventUpdate

    triggered_id = ctx.triggered[0]["prop_id"].split(".")[0]

    # Case 1: refine button was clicked
    if triggered_id == "refine-button":
        
        x0, y0, z0, roll0, pitch0, yaw0 = rot_trans_from_matrix(A_base)
        A_user = matrix_from_rot_trans(
            x0 + x_slider,
            y0 + y_slider,
            z0 + z_slider,
            roll0 + math.radians(roll_slider),
            pitch0 + math.radians(pitch_slider),
            yaw0 + math.radians(yaw_slider),
        )
        

        A_refined, _ = aligner.refine_matrix(
            A_cam_ant_init=A_user,
            ant_scale_init=s_base,
            camera_points=aligner.point_camera_locations,
            change_scale=False,
        )
        x, y, z, roll, pitch, yaw = rot_trans_from_matrix(A_refined)
        alignment_results.update({
            "x": x, "y": y, "z": z,
            "roll": roll, "pitch": pitch, "yaw": yaw,
        })

        x0, y0, z0, roll0, pitch0, yaw0 = rot_trans_from_matrix(A_base)
        dx = x - x0
        dy = y - y0
        dz = z - z0
        droll = (roll - roll0) * 180 / math.pi
        dpitch = (pitch - pitch0) * 180 / math.pi
        dyaw = (yaw - yaw0) * 180 / math.pi

        return (
        dx, dy, dz,
        droll, dpitch, dyaw,
        dx, dy, dz,
        droll, dpitch, dyaw,
        )

    # Case 2: input fields triggered the change
    elif "input" in triggered_id:
        return (
            x_input, y_input, z_input,
            roll_input, pitch_input, yaw_input,
            x_input, y_input, z_input,
            roll_input, pitch_input, yaw_input,
        )

    # Case 3: slider was moved directly
    else:
        return (
            x_slider, y_slider, z_slider,
            roll_slider, pitch_slider, yaw_slider,
            x_slider, y_slider, z_slider,
            roll_slider, pitch_slider, yaw_slider,
        )


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

    return f"Alignment values saved for {result['Specimen-Number']}. You may now close the window."




if __name__ == "__main__":
    app.run(debug=False)
