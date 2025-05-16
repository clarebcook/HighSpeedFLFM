# This GUI is used to manually match points between the three multi-perspective images
# 2024/11/12
# more cleaning needs to be done before this should be used as more than a demo script

# import
from hsflfm.util import MetadataManager, load_dictionary, save_dictionary
from hsflfm.calibration import FLF_System, generate_normalized_shift_maps

import os
import cv2
import sys
import torch
import torch.nn.functional as F
import numpy as np

import qtpy.QtWidgets as QtWidgets
import qtpy.QtGui as QtGui
from qtpy.QtCore import Qt

# specify specimen name
specimen_number = "20250429_OB_1"
data_manager = MetadataManager(specimen_number=specimen_number)

# specify if we're selecting alignment points or paint dots
point_type = "paint"
type_list = ["alignment", "paint"]

if point_type not in type_list:
    raise ValueError(f"point type must be one of {type_list}, not {point_type}")

# specify where this will be saved
save_folder = data_manager.alignment_folder
# right now, we anticipate these have already been made
assert os.path.exists(save_folder)
# if not os.path.exists(save_folder):
#    os.mkdir(save_folder)

if point_type == "alignment":
    name = "alignment_points"
    point_types = [
        "head_base",
        "eye_tipe",
        "under_eye_ridge",
        "ridge_top",
        "eye_back_tip",
    ]
elif point_type == "paint":
    name = "match_points"
    point_types = None
# adding demo right now to avoid overwriting past data
save_name = save_folder + f"/{name}"  # + "_demo"

# prepare the maps
calibration_filename = data_manager.calibration_filename
assert os.path.exists(calibration_filename)
system = FLF_System(calibration_filename)
info_manager = system.calib_manager
image_shape = info_manager.image_shape

warped_ss_maps = generate_normalized_shift_maps(
    calibration_filename=calibration_filename,
    image_shape=image_shape,
    gen_downsample=1,
    type="warped_shift_slope",
)
inv_inter_camera_maps = generate_normalized_shift_maps(
    calibration_filename=calibration_filename,
    image_shape=image_shape,
    gen_downsample=1,
    type="inv_inter_camera",
)


# prepare the shift and sum volume
# this function is defined here for now, since it is not needed elsewhere
def generate_warp_volume(image, heights, warped_shift_slopes, inv_inter_camera_map):
    # prepare the base grid
    y, x = torch.meshgrid(
        torch.arange(0, image_shape[0], dtype=torch.float32) / image_shape[0],
        torch.arange(0, image_shape[1], dtype=torch.float32) / image_shape[1],
    )
    base_grid = torch.stack([x, y], dim=-1)
    base_grid = base_grid * 2 - 1

    # make the grid stack with inter camera shifts
    base_grid = base_grid + inv_inter_camera_map
    base_grid = torch.stack([base_grid.squeeze(0)] * len(heights), dim=0)
    # make the slope shifts for each height
    heights = heights.view(-1, 1, 1, 1)
    slope_shifts = (
        torch.stack([warped_shift_slopes.squeeze(0)] * len(heights), dim=0) * heights
    )

    # add them
    # recall that the shift slopes were warped, but never multiplied by -1 at this stage
    # so we're doing that here
    grid = base_grid + slope_shifts * -1

    # then prepare the image
    image_stack = torch.stack([image.squeeze(0)] * len(heights), dim=0)

    warped_stack = F.grid_sample(
        image_stack, grid, mode="bilinear", padding_mode="zeros", align_corners=False
    )

    return warped_stack, grid


# load in the images
if point_type == "alignment":
    images = data_manager.light_calibration_images
elif point_type == "paint":
    images = data_manager.get_start_images(strike_number=1)

heights = torch.linspace(-3, 3, 200, dtype=torch.float32)
volume = torch.zeros((len(heights), images[0].shape[0], images[0].shape[1], 3))
grids = torch.zeros(
    (len(images), len(heights), images[0].shape[0], images[0].shape[1], 2)
)
for cam_num, image in images.items():
    image = torch.from_numpy(image[None, None]).to(torch.float32)

    ss_map = warped_ss_maps[[cam_num]].to(torch.float32)
    ii_map = inv_inter_camera_maps[[cam_num]].to(torch.float32)
    warp_volume, grid = generate_warp_volume(image, heights, ss_map, ii_map)
    warp_volume = warp_volume.squeeze()
    grid = grid.squeeze()

    color_index = cam_num % 3
    volume[:, :, :, color_index] += warp_volume
    grids[cam_num] = grid

# contrast adjust ?
volume = (volume - torch.min(volume)) / (torch.max(volume) - torch.min(volume)) * 255
volume = volume.to(torch.uint8)

# need to clean this up
# I can't remember why I was starting as tensors and converting to numpy
volume = volume.numpy()
heights = heights.numpy()
grids = grids.numpy()


class FrameViewer(QtWidgets.QWidget):
    def __init__(self, save_name, system, volume, heights, grid_volume, point_types):
        super().__init__()
        self.heights = heights
        self.volume = volume
        self.grid_volume = grid_volume
        self.current_frame = 0
        self.point_types = point_types
        self.point_number = 0
        self.save_name = save_name

        self.system = system
        self.info_manager = system.calib_manager
        camera_numbers = info_manager.image_numbers
        if os.path.exists(save_name):
            self.match_points = load_dictionary(save_name)
            self.add_point_to_volume(self.match_points)
        else:
            self.match_points = {}
            for num in camera_numbers:
                self.match_points[num] = []

        self.initUI()

    def initUI(self):
        layout = QtWidgets.QVBoxLayout()

        self.instruction_label = QtWidgets.QLabel()
        layout.addWidget(self.instruction_label)
        if self.point_types is not None:
            self.instruction_label.setText(self.point_types[self.point_number])

        self.graphics_view = QtWidgets.QGraphicsView()
        self.scene = QtWidgets.QGraphicsScene()
        self.graphics_view.setScene(self.scene)
        self.graphics_view.mouseDoubleClickEvent = self.on_double_click
        layout.addWidget(self.graphics_view)

        self.slider = QtWidgets.QSlider()
        self.slider.setOrientation(Qt.Horizontal)
        self.slider.setMinimum(0)
        self.slider.setMaximum(self.volume.shape[0] - 1)
        self.slider.setValue(0)
        self.slider.valueChanged.connect(self.on_slider_change)
        layout.addWidget(self.slider)

        self.height_label = QtWidgets.QLabel()
        layout.addWidget(self.height_label)

        self.setLayout(layout)

        self.update_frame()

    def update_frame(self):
        frame_data = self.volume[self.current_frame]
        frame_image = QtGui.QImage(
            frame_data.data,
            frame_data.shape[1],
            frame_data.shape[0],
            frame_data.shape[1] * 3,
            QtGui.QImage.Format_RGB888,
        )

        pixmap = QtGui.QPixmap.fromImage(frame_image)
        self.scene.clear()
        self.scene.addPixmap(pixmap)

        self.height_label.setText(f"height: {self.heights[self.current_frame]} mm")

        # Adjust the view
        self.graphics_view.fitInView(self.scene.sceneRect(), Qt.KeepAspectRatio)

    def on_slider_change(self, value):
        self.current_frame = value
        self.update_frame()

    def add_point_to_volume(self, match_points):
        # now we want to see where these points would appear throughout the volume...
        # I think we can just do a slope thing ?
        for key, point_sets in match_points.items():
            for points in point_sets:
                x_cam_pix = points[0]
                y_cam_pix = points[1]
                z_mm = points[2]
                x_vol_pix = points[3]
                y_vol_pix = points[4]
                ss = system.get_shift_slopes(key, [x_cam_pix], [y_cam_pix])

                for frame, height in zip(self.volume, self.heights):
                    locx = int(x_vol_pix + (height - z_mm) * ss[0])
                    locy = int(y_vol_pix + (height - z_mm) * ss[1])
                    color = [0, 0, 0]
                    color[key] = 255
                    cv2.circle(frame, (locy, locx), radius=1, thickness=1, color=color)
            # can remove this to show point selected in all three images
            break

    def on_double_click(self, event):
        pos = self.graphics_view.mapToScene(event.pos())

        # flip from display coordinates to array coordinates that we're used to
        # that's why we put x in for y etc
        x_vol_pix = int(pos.y())
        y_vol_pix = int(pos.x())
        z_mm = self.heights[self.current_frame]
        print(f"Double-clicked at coordinates: ({x_vol_pix}, {y_vol_pix})")

        for cam_num, volume in enumerate(self.grid_volume):
            shift_map = volume[self.current_frame]
            # get the normalized x and y values
            y_cam_norm, x_cam_norm = shift_map[x_vol_pix, y_vol_pix]
            # convert to pixels
            x_cam_pix = (x_cam_norm + 1) / 2 * image_shape[0]
            y_cam_pix = (y_cam_norm + 1) / 2 * image_shape[1]

            values = [
                float(i) for i in [x_cam_pix, y_cam_pix, z_mm, x_vol_pix, y_vol_pix]
            ]
            self.match_points[cam_num].append(values)

        save_dictionary(self.match_points, self.save_name)
        self.add_point_to_volume(self.match_points)

        self.point_number = self.point_number + 1
        if self.point_types is not None:
            self.instruction_label.setText(self.point_types[self.point_number])

        self.update_frame()


if __name__ == "__main__":
    app = QtWidgets.QApplication(sys.argv)
    viewer = FrameViewer(
        save_name=save_name,
        system=system,
        volume=volume,
        heights=heights,
        grid_volume=grids,
        point_types=point_types,
    )
    viewer.show()
    sys.exit(app.exec_())
