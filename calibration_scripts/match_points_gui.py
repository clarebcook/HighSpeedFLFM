# This GUI is used to manually match points between the three multi-perspective images
# 2024/11/12
# more cleaning needs to be done before this should be used as more than a demo script

# import
from hsflfm.util import MetadataManager, load_dictionary, save_dictionary
from hsflfm.calibration import FLF_System, generate_ss_volume

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
specimen_number = "20240506_OB_6"
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

# Load in Calibration settings and initialize info_manager and image_shape
calibration_filename = data_manager.calibration_filename
assert os.path.exists(calibration_filename)
system = FLF_System(calibration_filename)
info_manager = system.calib_manager
image_shape = info_manager.image_shape


# load in the images
if point_type == "alignment":
    images = data_manager.light_calibration_images
elif point_type == "paint":
    images = data_manager.get_start_images(strike_number=1)

# Define height range
heights = torch.linspace(-3, 3, 200, dtype=torch.float32)

# Generate the volume using the simplified, modular approach
volume, grids = generate_ss_volume(
    calibration_filename=data_manager.calibration_filename,
    images=images,
    heights=heights,
)

# Normalize and convert to uint8(similar to manual_strike_transfer)
volume = (volume - torch.min(volume)) / (torch.max(volume) - torch.min(volume)) * 255
volume = volume.to(torch.uint8)
volume = volume.numpy()


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


    # Overlaying circles on image   
    def image_to_volume_pixel(self, point, camera_number, height):
        x_pix, y_pix = point[0], point[1]
        ss = self.system.get_shift_slopes(camera_number, [x_pix], [y_pix])
        ii = self.system.get_pixel_shifts(camera_number, [x_pix], [y_pix])

        x_pix += float(height) * ss[0][0]
        y_pix += float(height) * ss[1][0]
        ii = self.system.get_pixel_shifts(camera_number, [x_pix], [y_pix])
        x_pix += ii[0][0]
        y_pix += ii[1][0]

        return x_pix, y_pix


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
        ## Draw ellipses on top of the image for each point

        # get the height value of the current frame
        z_mm = self.heights[self.current_frame]

        # check if match_points is not empty
        if self.match_points:
            # get the first camera number from match_points(to visualize points for one camera view)
            first_cam_num = system.reference_camera
            # Iterate through all the saved match points associated with that canera
            for point in self.match_points[first_cam_num]:
                # Convert image space pixel coordinates to volume-space pixel coordinates at current height
                x_pix, y_pix = self.image_to_volume_pixel(point, first_cam_num, z_mm)
                # Draw a circle on top of the current image at collected coords
                self.scene.addEllipse(y_pix, x_pix, 2.0, 2.0, QtGui.QPen(Qt.red))

        self.height_label.setText(f"height: {self.heights[self.current_frame]} mm")

        # Adjust the view
        self.graphics_view.fitInView(self.scene.sceneRect(), Qt.KeepAspectRatio)

    def on_slider_change(self, value):
        self.current_frame = value
        self.update_frame()

        
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
