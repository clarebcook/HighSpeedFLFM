from hsflfm.config import home_directory
from hsflfm.calibration import CalibrationInfoManager
from hsflfm.util import load_graph_images, MetadataManager
from hsflfm.calibration.vertices_organizing_functions import (
    organize_by_axis, ##My Import
    get_average_spacing,
    find_irregular_points
)

import numpy as np
from matplotlib import pyplot as plt
import cv2

# set "current_plane" and "current_camera_index" to values other than 0
# to start with plane/image other than the first one
# this can be done if this step is partially completed
global current_plane, current_camera_index, current_image, current_camera
current_plane = 0
current_camera_index = 0
current_image_set = None
current_image = None

# select any specimen filmed under the calibration conditions 
# alternatively, the calibration folder and filename can be manually specified 
specimen = "20240506_OB_6"
#specimen = "20250212_OB_1"
mm = MetadataManager(specimen)
calibration_filename = mm.calibration_filename
calibration_folder = mm.calibration_folder
calibration_manager = CalibrationInfoManager(calibration_filename)

vertices_dict = calibration_manager.all_vertices
if not vertices_dict:
    raise ValueError("You have not yet identified vertices for this image folder.")

plane_numbers = calibration_manager.plane_numbers
image_numbers = calibration_manager.image_numbers


def get_title():
    return f"Double click to remove point from camera {current_camera}, plane {current_plane} \n or double click out of canvas to proceed to next image"


def add_circles(irregular_points=None):
    image = current_image.copy()

    if len(image.shape) == 2 or image.shape[2] == 1:
        image = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)

    points = vertices_dict[current_plane][current_camera]

    regular_color = (128, 0, 0)       # Muted color: blue
    irregular_color = (0, 0, 255)     # Bright color: red
    radius = 3
    thickness = 1

    # Convert to set of rounded tuples
    irregular_set = set(map(tuple, np.round(irregular_points, 4))) if irregular_points is not None else set()

    for point in points:
        pt = tuple(np.round(point, 4))
        if pt in irregular_set:
            color = irregular_color
        else:
            color = regular_color
        cv2.circle(image, (int(point[0]), int(point[1])), radius, color, thickness)

    return image


def remove_point(event):
    if not event.dblclick:
        return
    global current_plane
    global current_camera
    global current_image

    ix, iy = event.xdata, event.ydata

    if ix is None or iy is None:
        # go on to the next image
        if current_camera == image_numbers[-1]:
            current_camera = 0
            current_plane = current_plane + 1

            if current_plane == plane_numbers[-1] + 1:
                calibration_manager.save_all_info()
                plt.close()
                return
        else:
            current_camera = current_camera + 1
        current_image = load_graph_images(
            folder=calibration_folder,
            image_numbers=[current_camera],
            plane_numbers=[current_plane],
            calibration_filename=calibration_filename,
        )[0][current_camera]

        # we'll take this moment to save progress
        calibration_manager.save_all_info()

    else:
        # remove the selected point
        vertices = vertices_dict[current_plane][current_camera]
        vertices = np.asarray(vertices)
        closest_index = np.argmin(
            np.sqrt((vertices[:, 0] - ix) ** 2 + (vertices[:, 1] - iy) ** 2)
        )
        vertices = np.delete(vertices, closest_index, axis=0)
        vertices_dict[current_plane][current_camera] = vertices.tolist()

    current_irregulars = all_irregulars.get((current_plane, current_camera), [])

    # Convert images so rgb codes work
    new_image = add_circles(irregular_points=current_irregulars)
    new_image_rgb = cv2.cvtColor(new_image, cv2.COLOR_BGR2RGB)
    im.set_data(new_image_rgb)

    fig.suptitle(get_title())
    fig.canvas.draw()
    return


fig, ax = plt.subplots(1, 1)
current_camera = image_numbers[current_camera_index]
current_image = load_graph_images(
    folder=calibration_folder,
    image_numbers=[current_camera],
    plane_numbers=[current_plane],
    calibration_filename=calibration_filename,
)[0][current_camera]


manager = CalibrationInfoManager(calibration_filename)
expected_spacing = manager.expected_line_spacing


# Initialize Irregulars Dictionary
all_irregulars = {}

# Loop through vertices dictionary and collect an array of irregular points
for plane_idx in vertices_dict:
    for view_idx in vertices_dict[plane_idx]:
        raw_points = vertices_dict[plane_idx][view_idx]

        # STEP 2: Filter only valid [x, y] points
        valid_points = [
            p for p in raw_points
            if isinstance(p, (list, tuple, np.ndarray)) and len(p) == 2 and all(isinstance(coord, (int, float)) for coord in p)
        ]

        verts = np.array(valid_points, dtype=np.float64)


        # Compute average spacing
        h_spacing, v_spacing = get_average_spacing(verts, expected_spacing = expected_spacing) # expected spacing from run_calibration

        # Organize rows and columns
        rows = organize_by_axis(verts, axis=1, expected_spacing=v_spacing) # Group points with similar y vals
        cols = organize_by_axis(verts, axis=0, expected_spacing=h_spacing) # Group points with similar x vals

        # Detect irregular points
        row_irregulars = find_irregular_points(rows, axis=0, spacing=h_spacing) 
        col_irregulars = find_irregular_points(cols, axis=1, spacing=v_spacing)

        # Collect points that are irregular in either axes
        final_irregulars = list(set(row_irregulars) | set(col_irregulars))

        if final_irregulars:  
            all_irregulars[(plane_idx, view_idx)] = final_irregulars

#Verification code(delete)
print(f"Irregular set contains {len(all_irregulars)} points.")


image = add_circles(irregular_points=all_irregulars.get((current_plane, current_camera), []))
image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
im = ax.imshow(image_rgb)
fig.suptitle(get_title())
cid = fig.canvas.mpl_connect("button_press_event", remove_point)

plt.show()


