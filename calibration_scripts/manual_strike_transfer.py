# this script can be used to manually match points between ant strikes
# to supplement points that are automatically matched

# 2024/12/03 this is heavily copied from "match_points_gui.py"
# which is easier for now, but these could be combined in the future

# This GUI is used to manually match points between the three multi-perspective images
# 2024/11/12
# more cleaning needs to be done before this should be used as more than a demo script

# import
from hsflfm.util import MetadataManager, load_dictionary, save_dictionary
from hsflfm.calibration import FLF_System, generate_normalized_shift_maps
from hsflfm.analysis import ResultManager

import sys
import torch
import numpy as np
import torch.nn.functional as F

import qtpy.QtWidgets as QtWidgets
import qtpy.QtGui as QtGui
from qtpy.QtCore import Qt


# prepare the shift and sum volume
# this function is defined here for now, since it is not needed elsewhere
def generate_warp_volume(
    image, heights, warped_shift_slopes, inv_inter_camera_map, image_shape
):
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

class FrameViewer(QtWidgets.QWidget):
    def __init__(
        self,
        alignment_info_filenames,
        heights,
    ):
        super().__init__()
        self.alignment_info_filenames = alignment_info_filenames
        self.cur_specimen_index = 0
        self.cur_strike_index = 0

        self.heights = heights
        self.current_frame = 0
        
        self.prepare_specimen()
        self.initUI()

    def initUI(self):
        layout = QtWidgets.QGridLayout()

        self.instruction_label = QtWidgets.QLabel()
        layout.addWidget(self.instruction_label, 0, 1)
        # if self.point_types is not None:
        #     self.instruction_label.setText(self.point_types[self.point_number])

        self.graphics_view = QtWidgets.QGraphicsView()
        self.scene = QtWidgets.QGraphicsScene()
        self.graphics_view.setScene(self.scene)
        self.graphics_view.mouseDoubleClickEvent = self.on_double_click
        layout.addWidget(self.graphics_view, 1, 1)

        self.slider = QtWidgets.QSlider()
        self.slider.setOrientation(Qt.Horizontal)
        self.slider.setMinimum(0)
        self.slider.setMaximum(self.volume.shape[0] - 1)
        self.slider.setValue(0)
        self.slider.valueChanged.connect(self.on_slider_change)
        layout.addWidget(self.slider, 2, 1)

        self.height_label = QtWidgets.QLabel()
        layout.addWidget(self.height_label, 3, 1)

        self.detail_label = QtWidgets.QLabel()
        layout.addWidget(self.detail_label, 0, 0)
        self.graphics_view_static = QtWidgets.QGraphicsView()
        self.scene_static = QtWidgets.QGraphicsScene()
        self.graphics_view_static.setScene(self.scene_static)
        self.graphics_view_static.mouseClickEvent = self.on_double_click

        layout.addWidget(self.graphics_view_static, 1, 0)

        self.skip_point_button = QtWidgets.QPushButton(text="Skip Point")
        layout.addWidget(self.skip_point_button, 2, 0)

        self.skip_point_button.clicked.connect(self.go_to_next_point)

        self.setLayout(layout)

        self.update_frame()

    def go_to_next_point(self):
        self.point_index = self.point_index + 1
        if self.point_index >= len(self.missing_points):
            self.go_to_next_specimen()
            return

        self.point_number = self.missing_points[int(self.point_index)]

        self.update_frame()

    def go_to_next_specimen(self):
        self.cur_specimen_index += 1
        if self.cur_specimen_index >= len(self.alignment_info_filenames):
            self.instruction_label.setText("All done!!!")
            return
        self.prepare_specimen()

    def prepare_specimen(self):
        self.cur_result_filename = self.alignment_info_filenames[
            self.cur_specimen_index
        ]
        info = load_dictionary(self.cur_result_filename)
        self.current_info = info
        self.current_info["manually_transferred_points"] = []

        # figure out which points are missing
        self.strike1_match_points = info["strike1_match_points"]

        self.point_numbers = np.asarray(info["point_numbers"])
        self.missing_points = np.asarray(
            list(
                set(np.arange(len(self.strike1_match_points[0])))
                - set(self.point_numbers)
            )
        )
        if len(self.missing_points) < 1:
            self.go_to_next_specimen()
            return
        self.point_index = 0
        self.point_number = self.missing_points[self.point_index]

        self.cur_result_manager = ResultManager(info)
        self.cur_specimen = info["specimen_number"]
        self.cur_strike = info["strike_number"]
        self.data_manager = MetadataManager(self.cur_specimen)
        images = self.data_manager.get_start_images(strike_number=self.cur_strike)

        calibration_filename = self.data_manager.calibration_filename
        self.system = FLF_System(calibration_filename)
        info_manager = self.system.calib_manager
        image_shape = info_manager.image_shape
        self.image_shape = image_shape
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

        volume = torch.zeros(
            (len(self.heights), images[0].shape[0], images[0].shape[1], 3)
        )
        self.grid_volume = torch.zeros(
            (3, len(self.heights), images[0].shape[0], images[0].shape[1], 2)
        )
        for cam_num, image in images.items():
            image = torch.from_numpy(image[None, None]).to(torch.float32)

            ss_map = warped_ss_maps[[cam_num]].to(torch.float32)
            ii_map = inv_inter_camera_maps[[cam_num]].to(torch.float32)
            warp_volume, grid = generate_warp_volume(
                image, self.heights, ss_map, ii_map, image_shape
            )
            warp_volume = warp_volume.squeeze()
            grid = grid.squeeze()

            volume[:, :, :, cam_num] = warp_volume
            self.grid_volume[cam_num] = grid
        volume = (
            (volume - torch.min(volume)) / (torch.max(volume) - torch.min(volume)) * 255
        )
        volume = volume.to(torch.uint8)
        volume = volume.numpy()
        self.volume = volume

        # get the existing match points for this strike
        self.match_points = info["match_points"]
        # self.add_point_to_volume(self.system, self.match_points)

        ref_cam = self.system.reference_camera
        self.strike1_image = self.data_manager.get_start_images(strike_number=1)[
            ref_cam
        ]

        for key, val in self.match_points.items():
            self.match_points[key] = np.asarray(val)
        # self.detail_label.setText(f"ant {self.cur_specimen}, ")

    def update_frame(self):
        # Step 1: update scene with the volume
        frame_data = self.volume[self.current_frame]
        frame_image = QtGui.QImage(
            frame_data.data,
            frame_data.shape[1],
            frame_data.shape[0],
            QtGui.QImage.Format_RGB888,
        )

        height = self.heights[self.current_frame]
        pixmap = QtGui.QPixmap.fromImage(frame_image)
        self.scene.clear()
        self.scene.addPixmap(pixmap)

        num = 2  # self.system.reference_camera
        mp = self.current_info["match_points"][num]
        new_points = self.current_info["manually_transferred_points"]
        pn = self.current_info["point_numbers"]
        for number, p in zip(pn, mp):
            # move to the correct spot
            x_pix = p[0]
            y_pix = p[1]
            ss = self.system.get_shift_slopes(num, [x_pix], [y_pix])
            ii = self.system.get_pixel_shifts(num, [x_pix], [y_pix])
            x_pix = x_pix + float(height) * ss[0][0]
            y_pix = y_pix + float(height) * ss[1][0]
            ii = self.system.get_pixel_shifts(num, [x_pix], [y_pix])
            x_pix = x_pix + ii[0][0]
            y_pix = y_pix + ii[1][0]

            if number in new_points:
                color = Qt.red
            else:
                color = Qt.black
            self.scene.addEllipse(y_pix, x_pix, 2, 2, QtGui.QPen(color))

        self.height_label.setText("height: {:.3f} mm".format(height))

        # Adjust the view
        self.graphics_view.fitInView(self.scene.sceneRect(), Qt.KeepAspectRatio)

        ## Step 2: ensure example image is up to date
        # for now feels easier to just make this a RGB image
        image = self.strike1_image[:, :, None].repeat(3, axis=-1)
        image = image.astype(np.uint8)
        img = QtGui.QImage(
            image,
            image.shape[1],
            image.shape[0],
            QtGui.QImage.Format_RGB888,
        )
        pixmap2 = QtGui.QPixmap.fromImage(img)
        self.scene_static.clear()
        self.scene_static.addPixmap(pixmap2)
        self.graphics_view_static.fitInView(
            self.scene_static.sceneRect(), Qt.KeepAspectRatio
        )
        for p in self.strike1_match_points[self.system.reference_camera]:
            self.scene_static.addEllipse(p[1], p[0], 3, 3, QtGui.QPen(Qt.blue))
        p = self.strike1_match_points[self.system.reference_camera][self.point_number]
        self.scene_static.addEllipse(p[1], p[0], 3, 3, QtGui.QPen(Qt.red))

    def on_slider_change(self, value):
        self.current_frame = value
        self.update_frame()

    def on_double_click(self, event):
        pos = self.graphics_view.mapToScene(event.pos())

        # flip from display coordinates to array coordinates that we're used to
        # that's why we put x in for y etc
        x_vol_pix = int(pos.y())
        y_vol_pix = int(pos.x())

        # this is currently only set up to work with three camreas
        # this will translate points back into space
        shift_map0 = self.grid_volume[0][self.current_frame]
        shift_map1 = self.grid_volume[1][self.current_frame]
        shift_map2 = self.grid_volume[2][self.current_frame]

        y_cam0_norm, x_cam0_norm = shift_map0[x_vol_pix, y_vol_pix]
        y_cam1_norm, x_cam1_norm = shift_map1[x_vol_pix, y_vol_pix]
        y_cam2_norm, x_cam2_norm = shift_map2[x_vol_pix, y_vol_pix]

        # convert to pixels
        x_cam0_pix = float((x_cam0_norm + 1) / 2 * self.image_shape[0])
        y_cam0_pix = float((y_cam0_norm + 1) / 2 * self.image_shape[1])
        x_cam1_pix = float((x_cam1_norm + 1) / 2 * self.image_shape[0])
        y_cam1_pix = float((y_cam1_norm + 1) / 2 * self.image_shape[1])
        x_cam2_pix = float((x_cam2_norm + 1) / 2 * self.image_shape[0])
        y_cam2_pix = float((y_cam2_norm + 1) / 2 * self.image_shape[1])

        if self.point_number in self.point_numbers:
            print(f"this point number {self.point_number} is already present: caution")
            return

        if (self.point_numbers > self.point_number).any():
            insert_index = np.where(self.point_numbers > self.point_number)[0][0]
        else:
            insert_index = len(self.point_numbers)
        self.point_numbers = np.insert(
            self.point_numbers, insert_index, self.point_number
        )

        for i, p in enumerate(
            [
                [x_cam0_pix, y_cam0_pix],
                [x_cam1_pix, y_cam1_pix],
                [x_cam2_pix, y_cam2_pix],
            ]
        ):

            self.match_points[i] = np.insert(
                self.match_points[i], insert_index, p, axis=0
            )

        self.current_info["point_numbers"] = self.point_numbers
        self.current_info["manually_transferred_points"].append(int(self.point_number))

        self.go_to_next_point()


if __name__ == "__main__":
    heights = torch.linspace(-3, 3, 200, dtype=torch.float32)
    app = QtWidgets.QApplication(sys.argv)
    viewer = FrameViewer(
        alignment_info_filenames=[
            "../temporary_result_storage_5/20240503_OB_3/strike_13_results.json",
            "../temporary_result_storage_5/20240503_OB_3/strike_14_results.json",
        ],
        heights=heights,
    )
    viewer.show()
    sys.exit(app.exec_())
