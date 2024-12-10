# this script can be used to manually match points between ant strikes
# to supplement points that are automatically matched

# 2024/12/03 this is heavily copied from "match_points_gui.py"
# which is easier for now, but these could be combined in the future

# This GUI is used to manually match points between the three multi-perspective images
# 2024/11/12
# more cleaning needs to be done before this should be used as more than a demo script

# import
from hsflfm.util import (
    MetadataManager,
    load_dictionary,
    save_dictionary,
    matmul,
    procrustes_analysis,
)
from hsflfm.calibration import (
    FLF_System,
    generate_ss_volume,
)  # generate_normalized_shift_maps
from hsflfm.analysis import ResultManager
from hsflfm.processing import Aligner, get_point_locations, world_frame_to_pixel

import sys
import torch
import numpy as np
import os
import torch.nn.functional as F

import qtpy.QtWidgets as QtWidgets
import qtpy.QtGui as QtGui
from qtpy.QtCore import Qt


class FrameViewer(QtWidgets.QWidget):
    def __init__(
        self,
        specimen_numbers,
        heights,
    ):
        super().__init__()
        self.specimen_numbers = specimen_numbers
        self.cur_specimen_index = 0
        self.strike_index = 0

        self.heights = heights
        self.current_frame = 0

        self.prepare_specimen()
        self.initUI()

        # TODO: there's likely a better way to do this?
        self.mode = "view"

    def initUI(self):
        # if this gets more complex, we'll load this from a .ui file
        # but this is okay for now
        layout = QtWidgets.QGridLayout()

        # set the instruction label
        self.instruction_label = QtWidgets.QLabel()
        layout.addWidget(self.instruction_label, 0, 1)

        # prepare the interactive view
        self.graphics_view = QtWidgets.QGraphicsView()
        self.scene = QtWidgets.QGraphicsScene()
        self.graphics_view.setScene(self.scene)
        self.graphics_view.mouseDoubleClickEvent = self.on_double_click
        layout.addWidget(self.graphics_view, 1, 1)

        # prepare the slider to change heights
        self.slider = QtWidgets.QSlider()
        self.slider.setOrientation(Qt.Horizontal)
        self.slider.setMinimum(0)
        self.slider.setMaximum(self.volume.shape[0] - 1)
        self.slider.setValue(0)
        self.slider.valueChanged.connect(self.on_slider_change)
        layout.addWidget(self.slider, 2, 1)

        self.height_label = QtWidgets.QLabel()
        layout.addWidget(self.height_label, 3, 1)

        # and buttons to approve, change transform, remove points, or re-run
        self.approve_button = QtWidgets.QPushButton(text="Approve")
        self.manual_button = QtWidgets.QPushButton(
            text="start new manual point transfer"
        )
        self.manual_button2 = QtWidgets.QPushButton(
            text="continue manual point transfer"
        )
        self.rerun_button = QtWidgets.QPushButton(text="Re-run alignment")
        # self.remove_points_button = QtWidgets.QPushButton(text="Remove points")
        for i, button in enumerate(
            [
                self.approve_button,
                self.manual_button,
                self.manual_button2,
                self.rerun_button,
                # self.remove_points_button,
            ]
        ):
            layout.addWidget(button, 4 + i, 1)
        self.approve_button.clicked.connect(self.approve)
        self.rerun_button.clicked.connect(self.rerun)
        self.manual_button.clicked.connect(self.prep_for_manual_alignment)
        self.manual_button2.clicked.connect(self.continue_manual_transfer)

        self.detail_label = QtWidgets.QLabel()
        layout.addWidget(self.detail_label, 0, 0)
        self.graphics_view_static = QtWidgets.QGraphicsView()
        self.scene_static = QtWidgets.QGraphicsScene()
        self.graphics_view_static.setScene(self.scene_static)
        self.graphics_view_static.mouseClickEvent = self.on_double_click

        layout.addWidget(self.graphics_view_static, 1, 0)

        self.skip_point_button = QtWidgets.QPushButton(text="Skip Point")
        layout.addWidget(self.skip_point_button, 2, 0)
        self.full_view_button = QtWidgets.QPushButton(text="View All")
        layout.addWidget(self.full_view_button, 3, 0)
        self.full_view_button.clicked.connect(self.view_all_points)

        self.skip_point_button.clicked.connect(self.go_to_next_point)

        self.setLayout(layout)

        self.update_frame()

    def view_all_points(self):
        self.mode = "view all"
        self.update_frame()

    def get_filename(self):
        # very hard-coded, we'll adjust this
        folder = "../temporary_alignment_data_storage"
        if not os.path.exists(folder):
            os.mkdir(folder)
        spec_folder = folder + f"/{self.cur_specimen_number}"
        if not os.path.exists(spec_folder):
            os.mkdir(spec_folder)
        filename = spec_folder + f"/strike_{self.strike_number}.json"
        return filename

    def approve(self):
        if self.mode != "view":
            print("cannot approve from this mode")
            return
        self.current_info["alignment_manually_modified"] = self.manually_modified
        if self.manually_modified:
            self.current_info["A_cam_to_ant_start"] = self.strike_A_cam_ant
            self.current_info["match_points"] = self.match_points
            self.current_info["point_numbers"] = self.point_numbers
            self.current_info["removed_points"] = []
            self.current_info["manual_point_numbers"] = self.manual_point_numbers

            # and update the aligner
            self.aligner.stored_alignment_matrices[self.strike_number] = (
                self.strike_A_cam_ant
            )
            self.aligner.stored_match_points[self.strike_number] = self.match_points
            self.aligner.stored_point_numbers[self.strike_number] = self.point_numbers

        filename = self.get_filename()
        save_dictionary(self.current_info, filename)

        self.go_to_next_strike()
        self.update_frame()

    def rerun(self):
        if self.mode != "add points":
            print("I don't know what to do with this yet")
            return
        self.manually_modified = True
        start_strike = self.current_info["aligned_from_strike_number"]
        prev_match_points = self.aligner.stored_match_points[start_strike]

        print("WARNING, NOT USING POINT INDICES CORRECTLY")
        # TODO: you need to use self.aligner.stored_point_numbers,
        # in case we start not using all the points at some point

        prev_strike_locations = get_point_locations(self.system, prev_match_points)
        strike_locations = get_point_locations(self.system, self.manual_match_points)
        # this is not ideal as its a copied/modified version of what's in aligner.py
        A_cam2_to_cam1, _, _ = procrustes_analysis(
            strike_locations,
            prev_strike_locations[self.manual_point_numbers],
            allow_scale=False,
        )

        all_strike_locations = matmul(
            np.linalg.inv(A_cam2_to_cam1), prev_strike_locations
        )
        strike_match_points = {}
        for camera in prev_match_points.keys():
            points = np.zeros((len(prev_match_points[camera]), 2))
            for i, p in enumerate(all_strike_locations):
                pixels = np.asarray(
                    world_frame_to_pixel(self.system, p, camera)
                ).squeeze()
                points[i] = pixels
            strike_match_points[camera] = points

        A_cam_ant = self.aligner.stored_alignment_matrices[start_strike]
        self.strike_A_cam_ant = np.linalg.matmul(A_cam_ant, A_cam2_to_cam1)

        # TODO: also handle here if start_strike didn't include all the numbers
        # update things...
        # point numbers might eventually need to be updated
        self.current_info["match_points"] = strike_match_points
        self.match_points = strike_match_points

        self.mode = "view"
        self.update_frame()
        return

    def continue_manual_transfer(self):
        self.mode = "add points"
        self.update_frame()

    def prep_for_manual_alignment(self):
        self.mode = "add points"

        # this can be cleaned up
        self.manual_match_points = {}
        for key in self.match_points.keys():
            self.manual_match_points[key] = []

        total_points = len(self.strike1_match_points[0])
        self.cycle_points = np.arange(total_points, dtype=np.uint16)
        np.random.shuffle(self.cycle_points)
        self.manual_point_numbers = np.asarray([])
        # self.current_info["manually_transferred_points"] = []
        # self.current_info["point_numbers"] = []
        # self.current_info["removed_points"] = []

        self.point_index = 0
        self.point_number = int(self.cycle_points[int(self.point_index)])
        self.update_frame()
        return

    def go_to_next_point(self):
        self.point_index = self.point_index + 1
        if self.point_index >= len(self.cycle_points):
            # do something to alert that there's no more points
            self.mode = "view"
            return

        self.point_number = int(self.cycle_points[int(self.point_index)])
        self.update_frame()

    def go_to_next_strike(self):
        self.strike_index += 1
        if self.strike_index >= len(self.strike_numbers):
            self.go_to_next_specimen()
            return

        self.prepare_strike()

    def go_to_next_specimen(self):
        # if this is the last specimen, do something
        self.cur_specimen_index += 1
        if self.cur_specimen_index >= len(self.specimen_numbers):
            # self.instruction_label.setText("All done!!!")
            # possibly do something else, like close the GUI
            return
        self.prepare_specimen()

    # we'll modify this...
    def load_past_information(self):
        folder = f"../temporary_alignment_data_storage/{self.cur_specimen_number}"
        if not os.path.exists(folder):
            return 0
        else:
            for index, num in enumerate(self.strike_numbers):
                filename = folder + f"/strike_{num}.json"
                if not os.path.exists(filename):
                    return index
                info = load_dictionary(filename)
                self.aligner.stored_alignment_matrices[num] = np.asarray(
                    info["A_cam_to_ant_start"]
                )
                for key, item in info["match_points"].items():
                    info["match_points"][key] = np.asarray(item)
                self.aligner.stored_match_points[num] = info["match_points"]
                self.aligner.stored_point_numbers[num] = np.asarray(
                    info["point_numbers"]
                )

        return -1

    def prepare_specimen(self):
        self.cur_specimen_number = self.specimen_numbers[self.cur_specimen_index]

        self.aligner = Aligner(
            self.cur_specimen_number,
            # alignment_settings={"enforce_stiff_transform": False},
        )
        self.data_manager = MetadataManager(self.cur_specimen_number)
        self.strike_numbers = np.sort(self.data_manager.strike_numbers)

        calibration_filename = self.data_manager.calibration_filename
        self.system = FLF_System(calibration_filename)
        self.strike1_image = self.data_manager.get_start_images(strike_number=1)[
            self.system.reference_camera
        ]

        self.strike_index = self.load_past_information()
        if self.strike_index == -1:
            self.go_to_next_specimen()
        self.prepare_strike()

    def prepare_strike(self):
        self.mode = "view"

        # perform the automatic alignment
        self.strike_number = self.strike_numbers[self.strike_index]

        self.current_info = self.aligner.prepare_strike_results(self.strike_number)
        info = self.current_info

        # info["manually_transferred_points"] = []

        # figure out which points are missing
        self.strike1_match_points = info["strike1_match_points"]
        self.point_numbers = np.asarray(info["point_numbers"])
        # self.missing_points = np.asarray(
        #     list(
        #         set(np.arange(len(self.strike1_match_points[0])))
        #         - set(self.point_numbers)
        #     )
        # )
        # if len(self.missing_points) < 1:
        #     print("no missing points: ", {len(self.missing_points)}, "on to next!")
        #     self.go_to_next_strike()
        #     return
        # self.point_index = 0
        # self.point_number = self.missing_points[self.point_index]

        # self.cur_result_manager = ResultManager(info)
        # self.cur_specimen = info["specimen_number"]
        # self.cur_strike = info["strike_number"]

        images = self.data_manager.get_start_images(strike_number=self.strike_number)
        self.image_shape = images[0].shape  # TODO: this is not good, do a different way

        volume, self.grid_volume = generate_ss_volume(
            self.data_manager.calibration_filename, images, self.heights
        )
        volume = (
            (volume - torch.min(volume)) / (torch.max(volume) - torch.min(volume)) * 255
        )
        volume = volume.to(torch.uint8)
        volume = volume.numpy()
        self.volume = volume

        # get the existing match points for this strike
        self.match_points = info["match_points"]
        self.manual_match_points = {}
        self.manual_point_numbers = np.asarray([])
        for key in self.match_points.keys():
            self.manual_match_points[key] = []
        self.strike_A_cam_ant = self.current_info["A_cam_to_ant_start"]
        self.manually_modified = False

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
        bad_points = self.current_info["removed_points"]
        pn = self.current_info["point_numbers"]
        if self.mode == "view":
            mp = self.current_info["match_points"][num]
            # new_points = self.current_info["manually_transferred_points"]

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

                if number in bad_points:
                    color = Qt.red
                elif number in self.manual_point_numbers:
                    color = Qt.green
                else:
                    color = Qt.black
                self.scene.addEllipse(y_pix, x_pix, 1, 1, QtGui.QPen(color))
        # repetitive... clean this up
        # ugh yeah this is very bad
        if self.mode == "view" or self.mode == "add points":
            mp = self.manual_match_points[num]
            pn = self.manual_point_numbers
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

                color = Qt.blue
                self.scene.addEllipse(y_pix, x_pix, 1, 1, QtGui.QPen(color))
        if self.mode == "view all":
            for num, mp in self.match_points.items():
                if num == 0:
                    color = Qt.red
                if num == 1:
                    color = Qt.green
                if num == 2:
                    color = Qt.blue
                for p in mp:
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

                    self.scene.addEllipse(y_pix, x_pix, 1, 1, QtGui.QPen(color))

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

        if self.mode == "add points":
            p = self.strike1_match_points[self.system.reference_camera][
                self.point_number
            ]
            self.scene_static.addEllipse(p[1], p[0], 1, 1, QtGui.QPen(Qt.red))
        else:
            for number, p in enumerate(
                self.strike1_match_points[self.system.reference_camera]
            ):
                if number in bad_points:
                    color = Qt.blue
                else:
                    color = Qt.black
                self.scene_static.addEllipse(p[1], p[0], 1, 1, QtGui.QPen(color))
            # p = self.strike1_match_points[self.system.reference_camera][self.point_number]
            # self.scene_static.addEllipse(p[1], p[0], 3, 3, QtGui.QPen(Qt.red))

        self.detail_label.setText(
            f"{self.cur_specimen_number}, analyzing strike {self.strike_number}"
        )

    def on_slider_change(self, value):
        self.current_frame = value
        self.update_frame()

    def on_double_click(self, event):
        if self.mode == "view" or self.mode == "view all":
            return

        elif self.mode == "add points":
            pos = self.graphics_view.mapToScene(event.pos())

            # flip from display coordinates to array coordinates that we're used to
            # that's why we put x in for y etc
            x_vol_pix = int(pos.y())
            y_vol_pix = int(pos.x())
            self.add_new_point(x_vol_pix, y_vol_pix)
            self.go_to_next_point()

    def add_new_point(self, x_vol_pix, y_vol_pix):
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

        if self.point_number in self.manual_point_numbers:
            print(f"this point number {self.point_number} is already present: caution")
            return

        if (self.manual_point_numbers > self.point_number).any():
            insert_index = np.where(self.manual_point_numbers > self.point_number)[0][0]
        else:
            insert_index = len(self.manual_point_numbers)
        self.manual_point_numbers = np.insert(
            self.manual_point_numbers, insert_index, self.point_number
        )

        for i, p in enumerate(
            [
                [x_cam0_pix, y_cam0_pix],
                [x_cam1_pix, y_cam1_pix],
                [x_cam2_pix, y_cam2_pix],
            ]
        ):
            if len(self.manual_match_points[i]) == 0:
                assert insert_index == 0
                self.manual_match_points[i] = np.asarray([p])
            else:
                self.manual_match_points[i] = np.insert(
                    self.manual_match_points[i], insert_index, p, axis=0
                )

        self.manual_point_numbers = self.manual_point_numbers.astype(np.uint16)
        # self.current_info["point_numbers"] = self.point_numbers
        # self.current_info["manually_transferred_points"].append(int(self.point_number))


if __name__ == "__main__":
    heights = torch.linspace(-3, 3, 200, dtype=torch.float32)
    app = QtWidgets.QApplication(sys.argv)
    viewer = FrameViewer(
        # alignment_info_filenames=[
        #    "../temporary_result_storage_5/20240503_OB_3/strike_13_results.json",
        #    "../temporary_result_storage_5/20240503_OB_3/strike_14_results.json",
        # ],
        specimen_numbers=["20240503_OB_3"],
        heights=heights,
    )
    viewer.show()
    sys.exit(app.exec_())
