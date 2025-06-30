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
        save_folder,
        demo_mode=False
    ):
        super().__init__()
        self.specimen_numbers = specimen_numbers
        self.cur_specimen_index = 0
        self.strike_index = 0

        self.heights = heights
        self.current_frame = 0

        self.basesavefolder = save_folder # This code to save folder location should go here. 
        self.demo_mode = demo_mode

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

        if self.demo_mode:
            demo_banner = QtWidgets.QLabel("DEMO MODE: No data will be saved")
            demo_banner.setStyleSheet("color: orange; font-weight: bold; font-size: 14px")
            layout.addWidget(demo_banner, 0, 1)
        else:
            demo_banner = QtWidgets.QLabel(self.basesavefolder)
            demo_banner.setStyleSheet("color: orange; font-weight: bold; font-size: 14px")
            layout.addWidget(demo_banner, 0, 1)


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
            text="Start new manual point transfer"
        )
        self.manual_button2 = QtWidgets.QPushButton(
            text="Continue manual point transfer"
        )
        self.rerun_button = QtWidgets.QPushButton(text="Re-run alignment")
        # self.remove_points_button = QtWidgets.QPushButton(text="Remove points")

        # GUI Functionality
        self.detail_label = QtWidgets.QLabel()
        self.graphics_view_static = QtWidgets.QGraphicsView()
        self.scene_static = QtWidgets.QGraphicsScene()
        self.graphics_view_static.setScene(self.scene_static)
        self.graphics_view_static.mouseClickEvent = self.on_double_click

        self.skip_point_button = QtWidgets.QPushButton(text="Skip Point")
        self.full_view_button = QtWidgets.QPushButton(text="View All")
        self.add_missing_points_button = QtWidgets.QPushButton(text="Add Missing Points")
        self.remove_points_button = QtWidgets.QPushButton(text="Remove Points")

        # Connect Buttons
        self.approve_button.clicked.connect(self.approve)
        self.rerun_button.clicked.connect(self.rerun)
        self.manual_button.clicked.connect(self.prep_for_manual_alignment)
        self.manual_button2.clicked.connect(self.continue_manual_transfer)
        self.full_view_button.clicked.connect(self.view_all_points)
        self.skip_point_button.clicked.connect(self.skip_point)
        self.add_missing_points_button.clicked.connect(self.add_missing_points)
        self.remove_points_button.clicked.connect(self.remove_points)

        # Button Formatting
        self.full_view_button.setStyleSheet("background-color: #6A5ACD; font-weight: bold;")
        self.add_missing_points_button.setStyleSheet("background-color: #6A5ACD; font-weight: bold;")
        self.remove_points_button.setStyleSheet("background-color: #6A5ACD; font-weight: bold;")
        self.manual_button.setStyleSheet("background-color: #6A5ACD; font-weight: bold;")
        self.approve_button.setStyleSheet("background-color: #2E8B57; color: white; font-weight: bold;")

        # GUI Layout
        layout.addWidget(self.detail_label, 0, 0)
        layout.addWidget(self.graphics_view_static, 1, 0)
        layout.addWidget(self.full_view_button, 2, 0)
        layout.addWidget(self.add_missing_points_button, 3, 0)
        layout.addWidget(self.remove_points_button, 4, 0)
        layout.addWidget(self.manual_button, 5, 0)
        layout.addWidget(self.approve_button, 6, 0)

        layout.addWidget(self.skip_point_button, 4, 1)
        layout.addWidget(self.manual_button2, 5, 1)
        layout.addWidget(self.rerun_button, 6, 1)

        self.setLayout(layout)

        self.update_frame()

    def view_all_points(self):
        self.mode = "view all"
        self.update_frame()
        self.update_button_states()

    def get_filename(self):
        # very hard-coded, we'll adjust this
        folder = self.basesavefolder  # test_results_from_manual_strike_transfer_GUI"
        if not os.path.exists(folder):
            os.mkdir(folder)
        spec_folder = folder + f"/{self.cur_specimen_number}"
        if not os.path.exists(spec_folder):
            os.mkdir(spec_folder)
        filename = spec_folder + f"/strike_{int(self.strike_number)}.json"
        return filename

    def approve(self):
        if self.mode == "manual align":
            print("cannot approve from this mode")
            return
        self.current_info["alignment_manually_modified"] = self.manually_modified
        if True:  # self.manually_modified:
            self.recompute_matrix()
            self.current_info["A_cam_to_ant_start"] = self.strike_A_cam_ant
            self.current_info["match_points"] = self.match_points
            self.current_info["point_numbers"] = self.point_numbers
            self.current_info["removed_points"] = self.manual_removed_points
            self.current_info["manual_align_point_numbers"] = (
                self.manual_align_point_numbers
            )

            # and update the aligner
            self.aligner.stored_alignment_matrices[self.strike_number] = (
                self.strike_A_cam_ant
            )
            # need to make sure the values aren't lists?
            for key, item in self.match_points.items():
                self.match_points[key] = np.asarray(item)
            self.aligner.stored_match_points[self.strike_number] = self.match_points
            self.aligner.stored_point_numbers[self.strike_number] = self.point_numbers

        filename = self.get_filename()

        if not self.demo_mode:
            save_dictionary(self.current_info, filename)
            print(f"Saved alignment results to {filename}")
        else:
            print("DEMO MODE: Skipped saving results")

        # save_dictionary(self.current_info, filename)
        # print(f"PAY ATTENTION saving as {filename}")
        # # print("WARNING WARNING NOT SAVING")

        self.go_to_next_strike()
        self.update_frame()

    def recompute_matrix(self):
        key = 1
        prev_match_points = self.aligner.stored_match_points[key]
        prev_matrix = self.aligner.stored_alignment_matrices[key]
        prev_locations = get_point_locations(self.system, prev_match_points)
        locations = get_point_locations(self.system, self.match_points)

        # figure out matching point numbers
        prev_point_numbers = self.aligner.stored_point_numbers[key]
        point_numbers = self.point_numbers
        shared_pn = np.intersect1d(prev_point_numbers, point_numbers)
        prev_indices = np.where(np.isin(prev_point_numbers, shared_pn))
        indices = np.where(np.isin(point_numbers, shared_pn))

        prev_locations = prev_locations[prev_indices]
        locations = locations[indices]

        A_cam2_to_cam1, _, _ = procrustes_analysis(
            locations, prev_locations, allow_scale=False
        )
        self.strike_A_cam_ant = np.matmul(prev_matrix, A_cam2_to_cam1)

    def rerun(self):
        if self.mode != "manual align":
            print("I don't know what to do with this yet")
            return
        self.manually_modified = True

        # update rough alignment for the current strike
        all_avg_diff = None
        for key in self.manual_align_match_points.keys():
            mp1 = self.aligner.stored_match_points[1][key][
                self.manual_align_point_numbers
            ]
            mp2 = self.manual_align_match_points[key]
            diff = mp2 - mp1
            avg_diff = np.mean(diff, axis=0)
            if all_avg_diff is None:
                all_avg_diff = avg_diff
            else:
                all_avg_diff += avg_diff
        all_avg_diff /= len(self.manual_align_match_points)

        # if this and the next strike started with the same rough alignment values
        # we'll update both
        update_next = False
        if self.strike_index < len(self.strike_numbers) - 1:
            next_strike_number = int(self.strike_numbers[self.strike_index + 1])
            cur_vals = self.aligner.rough_interstrike_alignment[int(self.strike_number)]
            next_vals = self.aligner.rough_interstrike_alignment[next_strike_number]
            if (
                abs(cur_vals[0] - next_vals[0]) < 1e-5
                and abs(cur_vals[1] - next_vals[1]) < 1e-5
            ):
                update_next = True

        print("I will be updating next")

        self.aligner.rough_interstrike_alignment[int(self.strike_number)] = all_avg_diff
        if update_next:
            self.aligner.rough_interstrike_alignment[next_strike_number] = all_avg_diff

        # TODO: decide if it makes sense to provide a start_strike here
        if self.cur_specimen_number == "20240506_OB_3" and self.strike_number == 3:
            start_strike = 1
        else:
            start_strike = None
        self.current_info = self.aligner.prepare_strike_results(
            self.strike_number, start_strike=start_strike, show=True
        )

        # TODO: this is heavily copied from "prepare_strike",
        # could make sense to consolidate
        info = self.current_info
        self.point_numbers = np.asarray(info["point_numbers"])
        self.match_points = info["match_points"]
        self.strike_A_cam_ant = info["A_cam_to_ant_start"]
        self.missing_points = np.asarray(
            list(
                set(np.arange(len(self.strike1_match_points[0])))
                - set(self.point_numbers)
            )
        )

        self.mode = "view"
        self.update_frame()
        return

        start_strike = self.current_info["aligned_from_strike_number"]
        prev_match_points = self.aligner.stored_match_points[start_strike]

        print("WARNING, NOT USING POINT INDICES CORRECTLY")
        # TODO: you need to use self.aligner.stored_point_numbers,
        # in case we start not using all the points at some point

        prev_strike_locations = get_point_locations(self.system, prev_match_points)
        strike_locations = get_point_locations(
            self.system, self.manual_align_match_points
        )
        # this is not ideal as its a copied/modified version of what's in aligner.py
        A_cam2_to_cam1, _, _ = procrustes_analysis(
            strike_locations,
            prev_strike_locations[self.manual_align_point_numbers],
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
        self.mode = "manual align"
        self.update_frame()

    def prep_for_manual_alignment(self):
        self.mode = "manual align"

        # this can be cleaned up
        self.manual_align_match_points = {}
        for key in self.match_points.keys():
            self.manual_align_match_points[key] = []

        total_points = len(self.strike1_match_points[0])
        self.cycle_points = np.arange(total_points, dtype=np.uint16)
        np.random.shuffle(self.cycle_points)
        self.manual_align_point_numbers = np.asarray([])
        # self.current_info["manually_transferred_points"] = []
        # self.current_info["point_numbers"] = []
        # self.current_info["removed_points"] = []

        self.point_index = 0
        self.point_number = int(self.cycle_points[int(self.point_index)])
        self.update_frame()
        self.update_button_states()
        return

    def add_missing_points(self):
        if len(self.missing_points) < 1:
            print("no missing points!")
            return

        self.point_index = 0
        self.mode = "add points"
        self.point_number = int(self.missing_points[int(self.point_index)])
        self.update_frame()
        self.update_button_states()
        return

    def remove_points(self):
        self.mode = "remove points"
        self.update_frame()
        self.update_button_states()
        return

    def skip_point(self):
        if self.mode not in ["add points", "manual align"]:
            print(f"not set up for skipping point in mode {self.mode}")
        self.point_index += 1
        self.go_to_next_point()

    def go_to_next_point(self):
        if self.mode == "manual align":
            self.point_index += 1
            points = self.cycle_points

            if self.point_index >= len(points):
                # do something to alert that there's no more points
                self.mode = "view"
                return
        elif self.mode == "add points":
            # we pretty much just stay at self.point_index = 0
            points = self.missing_points
            if len(points) <= self.point_index:
                self.mode = "view"
                return

        self.point_number = int(points[int(self.point_index)])
        self.update_frame()

    def go_to_next_strike(self):
        self.strike_index += 1
        if self.strike_index >= len(self.strike_numbers):
            self.go_to_next_specimen()
            return

        if (
            self.cur_specimen_number == "20240506_OB_3"
            and self.strike_numbers[self.strike_index] == 2
        ):
            self.strike_index += 1
        print(self.cur_specimen_number, self.strike_numbers[self.strike_index])
        self.prepare_strike()

    def go_to_next_specimen(self):
        self.cur_specimen_index += 1
        if self.cur_specimen_index >= len(self.specimen_numbers):
            print("All specimens processed. Closing GUI.")
            self.close()  
            return
        
        self.prepare_specimen()

    # we'll modify this...
    def load_past_information(self):
        return 0
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
            alignment_settings={
                "enforce_stiff_transform": False,
                "enforce_self_consistency": True,
            },
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

        if self.cur_specimen_number == "20240506_OB_3" and self.strike_number == 3:
            start_strike = 1
        else:
            start_strike = None
        self.current_info = self.aligner.prepare_strike_results(
            self.strike_number, start_strike=start_strike, show=self.strike_number == 1
        )

        # TODO: get rid of this again
        # if self.strike_number <= 8 and self.strike_number > 1:
        #    filename = f"../temp_redone_results/{self.cur_specimen_number}/strike_{self.strike_number}.json"
        #    self.current_info = load_dictionary(filename)
        #    print("WARNING LOADING PAST INFORMATION")

        filename = self.get_filename()
        if os.path.exists(filename):
            print("WARNING: LOADING PAST INFORMATION")
            self.current_info = load_dictionary(filename)
        else:
            print(f"{filename} not existing yet")

        info = self.current_info
        # info["manually_transferred_points"] = []

        # figure out which points are missing
        self.strike1_match_points = info["strike1_match_points"]
        self.point_numbers = np.asarray(info["point_numbers"])
        self.missing_points = np.asarray(
            list(
                set(np.arange(len(self.strike1_match_points[0])))
                - set(self.point_numbers)
            )
        )
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
        self.manual_align_match_points = {}
        self.manual_align_point_numbers = np.asarray([])
        for key in self.match_points.keys():
            self.manual_align_match_points[key] = []
        self.strike_A_cam_ant = self.current_info["A_cam_to_ant_start"]
        self.manually_modified = False
        self.manual_point_numbers = np.empty(0, dtype=int)

        self.manual_removed_points = np.empty(0, dtype=int)

    def update_frame(self):
        # can move this
        def image_to_volume_pixel(point, camera_number, height):
            x_pix = point[0]
            y_pix = point[1]
            ss = self.system.get_shift_slopes(camera_number, [x_pix], [y_pix])
            ii = self.system.get_pixel_shifts(camera_number, [x_pix], [y_pix])
            x_pix = x_pix + float(height) * ss[0][0]
            y_pix = y_pix + float(height) * ss[1][0]
            ii = self.system.get_pixel_shifts(camera_number, [x_pix], [y_pix])
            x_pix = x_pix + ii[0][0]
            y_pix = y_pix + ii[1][0]
            return x_pix, y_pix

        # Step 1: update scene with the volume
        frame_data = self.volume[self.current_frame]
        frame_image = QtGui.QImage(
            frame_data.data,
            frame_data.shape[1],
            frame_data.shape[0],
            frame_data.shape[1] * 3,
            QtGui.QImage.Format_RGB888,
        )

        height = self.heights[self.current_frame]
        pixmap = QtGui.QPixmap.fromImage(frame_image)
        self.scene.clear()
        self.scene.addPixmap(pixmap)

        num = 2  # self.system.reference_camera
        pn = self.point_numbers  # self.current_info["point_numbers"]
        if self.mode in ["view", "add points", "remove points"]:
            # mp = self.current_info["match_points"][num]
            mp = self.match_points[num]
            # new_points = self.current_info["manually_transferred_points"]

            for number, p in zip(pn, mp):
                # move to the correct spot
                x_pix, y_pix = image_to_volume_pixel(p, num, height)

                # if number in self.missing_points:
                #    color = Qt.red
                if number in self.manual_align_point_numbers:
                    color = Qt.green
                elif number in self.manual_point_numbers:
                    color = Qt.purple
                else:
                    color = Qt.black
                self.scene.addEllipse(y_pix, x_pix, 1, 1, QtGui.QPen(color))
        # repetitive... clean this up
        # ugh yeah this is very bad
        if self.mode == "view" or self.mode == "manual align":
            mp = self.manual_align_match_points[num]
            pn = self.manual_align_point_numbers
            for number, p in zip(pn, mp):
                x_pix, y_pix = image_to_volume_pixel(p, num, height)
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
                    x_pix, y_pix = image_to_volume_pixel(p, num, height)
                    self.scene.addEllipse(y_pix, x_pix, 1, 1, QtGui.QPen(color))

        self.height_label.setText("height: {:.3f} mm".format(height))

        # Adjust the view
        self.graphics_view.fitInView(self.scene.sceneRect(), Qt.KeepAspectRatio)

        ## Step 2: ensure example image is up to date
        # for now feels easier to just make this a RGB image
        image = self.strike1_image[:, :, None].repeat(3, axis=-1)
        image = (image - np.min(image)) / (np.max(image) - np.min(image)) * 255
        image = image.astype(np.uint8)
        img = QtGui.QImage(
            image,
            image.shape[1],
            image.shape[0],
            image.shape[1] * 3,
            QtGui.QImage.Format_RGB888,
        )
        pixmap2 = QtGui.QPixmap.fromImage(img)
        self.scene_static.clear()
        self.scene_static.addPixmap(pixmap2)
        self.graphics_view_static.fitInView(
            self.scene_static.sceneRect(), Qt.KeepAspectRatio
        )

        if self.mode in ["manual align", "add points"]:
            # self.aligner.stored_match_points[1][self.system.reference_camera][
            #    self.point_number
            # ]
            p = self.strike1_match_points[self.system.reference_camera][
                self.point_number
            ]
            self.scene_static.addEllipse(p[1], p[0], 1, 1, QtGui.QPen(Qt.red))
        else:
            for number, p in enumerate(
                self.strike1_match_points[self.system.reference_camera]
            ):
                if number in self.missing_points:
                    color = Qt.blue
                else:
                    color = Qt.black
                self.scene_static.addEllipse(p[1], p[0], 1, 1, QtGui.QPen(color))
            # p = self.strike1_match_points[self.system.reference_camera][self.point_number]
            # self.scene_static.addEllipse(p[1], p[0], 3, 3, QtGui.QPen(Qt.red))

        self.detail_label.setText(
            f"{self.cur_specimen_number}, analyzing strike {self.strike_number}, mode {self.mode}"
        )

        self.detail_label.setText(
            f'{self.cur_specimen_number}, analyzing strike: {self.strike_number}, mode: {self.mode}'
        )
        self.detail_label.setStyleSheet("font-weight: bold; font-size: 14px;")

    def on_slider_change(self, value):
        self.current_frame = value
        self.update_frame()

    def on_double_click(self, event):
        if self.mode == "view" or self.mode == "view all":
            return

        pos = self.graphics_view.mapToScene(event.pos())
        # flip from display coordinates to array coordinates that we're used to
        # that's why we put x in for y etc
        x_vol_pix = int(pos.y())
        y_vol_pix = int(pos.x())

        if self.mode == "manual align" or self.mode == "add points":
            self.add_new_point(x_vol_pix, y_vol_pix)
            self.go_to_next_point()
        if self.mode == "remove points":
            self.remove_point_at_position(x_vol_pix, y_vol_pix)
        self.update_frame()

    def _volume_pixel_to_image(self, x_vol_pix, y_vol_pix, camera_number, frame=None):
        if frame is None:
            frame = self.current_frame
        shift_map = self.grid_volume[camera_number][frame]
        y_norm, x_norm = shift_map[x_vol_pix, y_vol_pix]
        x_cam_pix = float((x_norm + 1) / 2 * self.image_shape[0])
        y_cam_pix = float((y_norm + 1) / 2 * self.image_shape[1])
        return x_cam_pix, y_cam_pix

    def remove_point_at_position(self, x_vol_pix, y_vol_pix):
        # identify the closest point
        cam_num = self.system.reference_camera
        x_cam_pix, y_cam_pix = self._volume_pixel_to_image(
            x_vol_pix, y_vol_pix, cam_num
        )
        cam_mp = np.asarray(self.match_points[cam_num])
        distances = np.linalg.norm(cam_mp - [x_cam_pix, y_cam_pix], axis=1)
        index = np.argmin(distances)

        point_number = self.point_numbers[index]
        self.point_numbers = np.delete(self.point_numbers, index)
        self.manual_removed_points = np.concatenate(
            (self.manual_removed_points, [point_number])
        )
        self.missing_points = np.concatenate((self.missing_points, [point_number]))

        for key, item in self.match_points.items():
            self.match_points[key] = np.delete(item, index, axis=0)

        return

    def add_new_point(self, x_vol_pix, y_vol_pix):
        pixels = np.zeros((len(self.system.calib_manager.image_numbers), 2))
        for i, cam_num in enumerate(self.system.calib_manager.image_numbers):
            x_cam_pix, y_cam_pix = self._volume_pixel_to_image(
                x_vol_pix, y_vol_pix, cam_num
            )
            pixels[i] = [x_cam_pix, y_cam_pix]

        # TODO: massively clean this up
        if self.mode == "manual align":
            if self.point_number in self.manual_align_point_numbers:
                print(
                    f"this point number {self.point_number} is already present: caution",
                    self.manual_align_point_numbers,
                )
                return

            if (self.manual_align_point_numbers > self.point_number).any():
                insert_index = np.where(
                    self.manual_align_point_numbers > self.point_number
                )[0][0]
            else:
                insert_index = len(self.manual_align_point_numbers)
            self.manual_align_point_numbers = np.insert(
                self.manual_align_point_numbers, insert_index, self.point_number
            )

            for i, p in enumerate(pixels):
                if len(self.manual_align_match_points[i]) == 0:
                    assert insert_index == 0
                    self.manual_align_match_points[i] = np.asarray([p])
                else:
                    self.manual_align_match_points[i] = np.insert(
                        self.manual_align_match_points[i], insert_index, p, axis=0
                    )

            self.manual_align_point_numbers = self.manual_align_point_numbers.astype(
                np.uint16
            )
        elif self.mode == "add points":
            if self.point_number in self.point_numbers:
                print(
                    f"this point number {self.point_number} is already present: caution"
                )
                return
            if (self.point_numbers > self.point_number).any():
                insert_index = np.where(self.point_numbers > self.point_number)[0][0]
            else:
                insert_index = len(self.point_numbers)

            self.point_numbers = np.insert(
                self.point_numbers, insert_index, self.point_number
            )

            for i, p in enumerate(pixels):
                if len(self.match_points[i]) == 0:
                    assert insert_index == 0
                    self.match_points[i] = np.asarray([p])
                else:
                    self.match_points[i] = np.insert(
                        self.match_points[i], insert_index, p, axis=0
                    )

            self.point_numbers = self.point_numbers.astype(np.uint16)
            remove_idx = np.where(self.missing_points == self.point_number)[0][0]
            self.missing_points = np.delete(self.missing_points, remove_idx)
            self.manual_points_numbers = np.concatenate(
                (self.manual_point_numbers, [self.point_number])
            )

        print("missing points: ", self.missing_points)
        print()

    def update_button_states(self):
        if self.mode == "add points":
            self.skip_point_button.setEnabled(True)
            self.approve_button.setEnabled(True)
            self.remove_points_button.setEnabled(True)
            self.manual_button.setEnabled(True)
            self.manual_button2.setEnabled(False)
            self.rerun_button.setEnabled(False)
            self.add_missing_points_button.setEnabled(True)
            self.full_view_button.setEnabled(True)
        elif self.mode == "view all":
            self.skip_point_button.setEnabled(False)
            self.approve_button.setEnabled(True)
            self.remove_points_button.setEnabled(True)
            self.manual_button.setEnabled(True)
            self.manual_button2.setEnabled(False)
            self.rerun_button.setEnabled(False)
            self.add_missing_points_button.setEnabled(True)
            self.full_view_button.setEnabled(True)
        elif self.mode == "remove points":
            self.skip_point_button.setEnabled(False)
            self.approve_button.setEnabled(True)
            self.remove_points_button.setEnabled(True)
            self.manual_button.setEnabled(True)
            self.manual_button2.setEnabled(False)
            self.rerun_button.setEnabled(False)
            self.add_missing_points_button.setEnabled(True)
            self.full_view_button.setEnabled(True)
        elif self.mode == "manual align":
            self.skip_point_button.setEnabled(True)
            self.approve_button.setEnabled(True)
            self.remove_points_button.setEnabled(True)
            self.manual_button.setEnabled(True)
            self.manual_button2.setEnabled(True)
            self.rerun_button.setEnabled(True)
            self.add_missing_points_button.setEnabled(True)
            self.full_view_button.setEnabled(True)
        else:  # default: view mode or any other mode
            self.skip_point_button.setEnabled(True)
            self.approve_button.setEnabled(True)
            self.remove_points_button.setEnabled(True)
            self.manual_button.setEnabled(True)
            self.manual_button2.setEnabled(True)
            self.rerun_button.setEnabled(False)
            self.add_missing_points_button.setEnabled(True)
            self.full_view_button.setEnabled(True)

## Change Settings Here

if __name__ == "__main__":
    heights = torch.linspace(-3, 3, 200, dtype=torch.float32)
    app = QtWidgets.QApplication(sys.argv)
    viewer = FrameViewer(
        # alignment_info_filenames=[
        #    "../temporary_result_storage_5/20240503_OB_3/strike_13_results.json",
        #    "../temporary_result_storage_5/20240503_OB_3/strike_14_results.json",
        # ],
        # specimen_numbers=["20250226_OB_2"],  # , "20240503_OB_3"],
        # specimen_numbers=["20250429_OB_1"],
        specimen_numbers=["20240506_OB_6"],
        heights=heights,
        save_folder="/Users/abhin/Documents/Graduate School/Patek Research Docs/Ant Strike Outputs", # don't give it a default folder
        demo_mode=False 
    )
    viewer.show()
    sys.exit(app.exec_())
