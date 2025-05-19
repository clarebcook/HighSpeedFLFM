from hsflfm.config import home_directory
from hsflfm.calibration import CalibrationInfoManager
from hsflfm.util import load_graph_images, MetadataManager

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

specimen = "20250429_OB_1"
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


def add_circles():
    image = current_image.copy()  # all_images[current_plane][current_camera].copy()
    points = vertices_dict[current_plane][current_camera]

    radius = 3  # 10
    color = (255, 0, 0)
    thickness = 1  # 4
    for point in points:
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

    im.set_data(add_circles())
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

image = add_circles()
im = ax.imshow(image)
fig.suptitle(get_title())
cid = fig.canvas.mpl_connect("button_press_event", remove_point)

plt.show()
