# this file takes points extracted from a CT scan of a trap jaw ant
# and uses them to establish the "ant coordinate system"

from hsflfm.util import estimate_plane, project_point_on_plane, matmul
import numpy as np

### coordinates of important points extracted from the CT scan STL file

# these are (x, y, z) points defining a shape containing stable points
stable_vertices_mesh = [
    [4814, 6592, 8946],
    [4928, 6217, 8933],
    [5062, 6031, 9111],
    [5183, 5919, 9313],
    [5263, 5889, 9479],
    [5313, 6036, 9709],
    [5320, 6200, 9851],
    [5160, 6703, 9851],
]

# and the x coordinate above which points are considered stable
saddle_ext_point_mesh = [5289, 4353, 8852]

# these points are all roughly along the midline of the ant's head
# this is a plane that should be parallel to our x-y plane
# the last two roughly define the x axis
midline_points_mesh = [
    [6949.597, 4620.833, 9169.048],
    [6641.815, 6306.41, 9545.366],
    [6925.743, 3898.059, 8363.759],
    [6280.361, 4740.525, 7545.067],
    [5974.011, 6170.237, 7895.482],
    [6741.965, 3920.49, 7984.114],
    [7002.05, 4643.4, 9135.56],
    [6646.277, 6277.355, 9539.609],
]

# locations of key points used in alignment
key_features_mesh = {
    "head_base": [5179.541, 6786.483, 8464.63],
    "eye_tip": [5497.405, 4289.137, 8871.049],
    "under_eye_ridge": [5252.716, 4494.765, 8974.761],
    "ridge_top": [5449.799, 5276.736, 9369.413],
    "eye_back_tip": [5616.102, 4852.72, 9363.777],
    "saddle_origin": [5213.325, 5505.217, 9070.891],
}

# relate these extracted values to our ant coordinate system

# identify the midline plane for the full ant head
midline_plane_coeffs = estimate_plane(midline_points_mesh)

# then shift the x-y plane to be on our chosen origin point for the coord system
origin = key_features_mesh["saddle_origin"]
proj = project_point_on_plane(origin, midline_plane_coeffs)

shift_vector = origin - proj

a, b, c, d = midline_plane_coeffs
new_d = d - a * shift_vector[0] - b * shift_vector[1] - c * shift_vector[2]

x_y_plane_coeffs = [a, b, c, new_d]

# then get our normalized x-axis
# first project the originally defined midline x axis onto this plane
# and shift down so it intersects with the origin


# the last two midline points define a vector which should be parallel to the x-axis on the midline plane
# so first we'll project those two points onto our new x-y plane
point1 = project_point_on_plane(midline_points_mesh[-1], x_y_plane_coeffs)
point2 = project_point_on_plane(midline_points_mesh[-2], x_y_plane_coeffs)

# then we can get a parallel line that originates at our defined origin
# and then normalize to get our normalized x-axis
x_axis_point = point2 - point1 + origin
norm = np.linalg.norm(x_axis_point - origin)
x_axis_point_n = x_axis_point - (x_axis_point - origin) / norm * (norm - 1)
# now, our new normalized x-axis will be between the origin and "x_axis_point_n"


# we can pretty much just use that already defined "shift_vector" as our z-axis, just needs to be normalized
shift_vector_norm = shift_vector / np.linalg.norm(shift_vector)
z_axis_point_n = shift_vector_norm + origin

# and the y axis is defined by the x and z
y_axis = np.cross(shift_vector_norm, x_axis_point_n - origin)
y_axis_point_n = y_axis + origin

# now that I have all my basis vectors, I should be able to create a transform matrix
v0 = x_axis_point_n - origin
v1 = y_axis_point_n - origin
v2 = z_axis_point_n - origin

A1 = np.empty((3, 3), dtype=np.float64)
A1[:, 0] = v0
A1[:, 1] = v1
A1[:, 2] = v2

A = np.zeros((4, 4), dtype=A1.dtype)
A[:3, :3] = A1
A[-1] = [0, 0, 0, 1]
A = np.linalg.inv(A)
for i in range(3):
    A[i, -1] = -(A[i][0] * origin[0] + A[i][1] * origin[1] + A[i][2] * origin[2])
M_mesh_ant = A

# and finally, find where your alignment points are in this coordinate frame
key_features_ant = {}
for key, p in key_features_mesh.items():
    p1 = p + [1]
    transformed = np.matmul(A, p1)[:-1]
    key_features_ant[key] = transformed

# and these key_features_ant are what will be imported elsewhere!

# then transform the points marking the stable regions
saddle_ext_point_ant = np.matmul(A, saddle_ext_point_mesh + [1])[:-1]

stable_vertices_ant = matmul(M_mesh_ant, stable_vertices_mesh)

# Finally, since we are using the opposite side of the head from where I grabbed the coords
# we need to flip the y axis to maintain a cartesian coordinate system
# with the z axis pointing out from the ant
for key, point in key_features_ant.items():
    key_features_ant[key] = [point[0], -point[1], point[2]]

stable_vertices_ant[:, 1] = -stable_vertices_ant[:, 1]
