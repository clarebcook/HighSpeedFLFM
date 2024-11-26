# description goes here

from hsflfm.config import home_directory
from hsflfm.ant_model import (
    mesh_filename,
    M_mesh_ant,
    key_features_ant,
    stable_vertices_ant,
)
from hsflfm.calibration import FLF_System
from hsflfm.util import (
    MetadataManager,
    load_dictionary,
    procrustes_analysis,
    matmul,
    matrix_from_rot_trans,
    rot_trans_from_matrix,
    cross_image
)

from .processing_functions import (
    get_point_locations,
    match_points_between_images,
    world_frame_to_pixel,
)

import trimesh
import cv2
import numpy as np
import functools
import scipy
from scipy.spatial import KDTree
from scipy.optimize import minimize
from shapely.geometry import Polygon, Point

# don't totally understand this number, but it's some sort of scaling difference between the
# mesh used here, and the original mesh where the alignment points were chosen
mesh_scale = 100

# default settings
default_alignment_settings = {
    "vertex_sample_density": 100,
    "refinement_bounds": [
        0.4,  # x (mm)
        0.4,  # y (mm)
        0.4,  # z (mm)
        0.26,  # roll (rad)
        0.26,  # pitch (rad)
        0.26,  # yaw (rad)
        200,  # scale (unitless)
    ],
    "scale_weight": 1e11,
    "huber_delta": 4000,
    "point_error_cutoff": 0.025,  # mm
    "use_init_scale": True,
    "base_align_point_names": [
        "head_base",
        "eye_tip",
        "under_eye_ridge",
        "ridge_top",
    ],  # , "eye_back_tip"]
    # for x, y, z, roll, pitch, yaw, scale
    "deviation_weights": np.asarray([5e1, 5e1, 5e1, 0, 0, 5e4, 5e-4]),
    "base_loss_weight": 1e-6,
    "point_match_sequentially": True, 
    "reshift_during_inter_strike_point_matching": True,
}

# individual start points
# for a few specific samples, the base alignment is far enough off
# that these start locations/angles were manually selected for fine refinement
# order is [x, y, z, roll, pitch, yaw, scale]
# NOTE: these are rotations/translations FROM ant TO camera
default_individual_base_alignment = {
    "20220422_OB_1": [3.0, 1.77, -0.366, 0.00353, -0.3525, -2.96, 1727.69627255],
    "20240502_OB_2": [
        2.819,
        1.647,
        -0.125,
        -0.120771,
        -0.31184,
        -2.9792,
        1867.32301648,
    ],
    "20240507_OB_2": [3.056, 1.825, -0.19818, -0.14607, -0.29805, 2.97, 1502.546848],
    "20240503_OB_3": [2.559, 2.3911, -0.4566, -0.13919, 0.2422, -3.075, 1619.64],
}

# this is to help with alignment between strikes 
# for a small number of ants, there was a very large shift between strikes
# thaa can be challenging to find automatically
# so it can be helpful to provide a rough alignment
# this can be cleaned up more later if necessary, but for now this impacts
# a small enough number of videos that this approach is fine 
default_rough_interstrike_alignment = {
    "20240507_OB_2": {
        1: (0, 0), 
        2: (0, 0), 
        3: (0, 0), 
        4: (0, 0), 
        5: (0, 0), 
        6: (0, 0), 
        7: (0, 0), 
        8: (-50, -15), 
        9: (-50, -15), 
        10:(-50, -15), 
        11:(-50, -15), 
        12:(-50, -15), 
        13:(-50, -15), 
        14:(-50, -15), 
        15:(-50, -15), 
        16:(-50, -15), 
    }
}

default_flow_settings = {
    "pyr_scale": 0.9,
    "levels": 5,
    "winsize": 21,
    "iterations": 5,
    "poly_n": 3,
    "poly_sigma": 0.8,
    "flags": 0,
}

individual_flow_settings = {
    "20240506_OB_1": {
        "poly_n": 7,
        "poly_sigma": 1.7
    },
    "20240417_OB_1": {
        "poly_n": 3, 
        "poly_sigma": 1.2
    },
    "20240418_OB_1": {
        "poly_n": 3,
        "poly_sigma": 1.2,
    }
}

save_keys = [
    "mesh_filename",
    "mesh_scale",
    "M_mesh_ant",
    "strike1_match_points",
    "match_points",
    "alignment_points",
    "key_feature_coordinates",
    "alignment_settings",
    "A_cam_to_ant_start",
    "A_cam_to_ant_start_strike1",
    "ant_scale",
    "stable_points",
    "point_numbers",
    "removed_points",
    "specimen_number",
    "strike_number",
    "base_alignment_values",
    # alignment is generally done sequentially with strikes, but if points are matched poorly
    # on one strike, we may go back a number for alignment. So that information
    # should be stored 
    "aligned_from_strike_number",
    "rough_interstrike_alignment",
    "strike_alignment_flow_settings"
   
]



# if "use_individual_base_alignment" is False, "base_alignment_values" won't be used
# if "individual_base_alignment" is None, the values will pull from "default_individual_base_alignment"
# and will just use the result of normal base alignment if values are not specified there
class Aligner:
    def __init__(
        self,
        specimen_name,
        alignment_settings={},
        use_individual_base_alignment=True,
        base_alignment_values=None,
        use_rough_interstrike_alignment=True,
    ):
        self.mesh_filename = mesh_filename
        self.mesh_scale = mesh_scale
        self.M_mesh_ant = M_mesh_ant
        self.mesh = trimesh.load(home_directory + "/" + self.mesh_filename)

        self.specimen_name = specimen_name

        self.data_manager = MetadataManager(specimen_number=specimen_name)

        self.match_points = load_dictionary(self.data_manager.match_points_filename)
        
        alignment_points = load_dictionary(self.data_manager.alignment_points_filename)

        self.system = FLF_System(self.data_manager.calibration_filename)

        self.alignment_settings = default_alignment_settings
        for key, value in alignment_settings.items():
            self.alignment_settings[key] = value

        if not use_individual_base_alignment:
            self.base_alignment_values = None
        elif (
            base_alignment_values is None
            and self.specimen_name in default_individual_base_alignment
        ):
            print("using manually specified pre-alignment")
            self.base_alignment_values = default_individual_base_alignment[
                self.specimen_name
            ]
        else:
            self.base_alignment_values = base_alignment_values

        # this should be cleaned up, but for now, we know the points are in a certain order
        # and we will keep only those
        self.alignment_points = {}
        features = self.alignment_settings["base_align_point_names"]
        for cam_num, points in alignment_points.items():
            new_points = np.zeros((len(features), 2))
            points = np.asarray(points)
            # j = 0
            for i, feature_name in enumerate(key_features_ant.keys()):
                new_idx = np.where(np.asarray(features) == feature_name)[0]
                if len(new_idx) < 1:
                    continue

                # if feature_name in features:
                new_points[new_idx[0]] = points[i, :2]
                # j = j + 1
            self.alignment_points[cam_num] = new_points

        # this can hold the match points from a range of videos
        # so we can theoretically track points sequentially between videos
        # the key will be the strike number
        p = self.match_points.copy() 
        for key, item in p.items():
            p[key] = np.asarray(item)[:, :2]
        self.stored_match_points = {1: p}
        self.stored_point_numbers = {1: np.arange(len(self.match_points[0]))}
        self.stored_alignment_matrices = {}

        self.stable_points = None
        self.A_cam_ant = None
        self.ant_scale = None

        if use_rough_interstrike_alignment and specimen_name in default_rough_interstrike_alignment:
            self.rough_interstrike_alignment = default_rough_interstrike_alignment[specimen_name]
        else:
            self.rough_interstrike_alignment = {} 
            for strike_num in self.data_manager.strike_numbers:
                self.rough_interstrike_alignment[strike_num] = (0, 0)

        self.flow_settings = default_flow_settings 
        if specimen_name in individual_flow_settings:
            ifs = individual_flow_settings[specimen_name]
            for key, item in ifs.items():
                self.flow_settings[key] = item

    @property
    def point_camera_locations(self):
        return get_point_locations(self.system, self.match_points)

    @property
    def key_feature_camera_locations(self):
        return get_point_locations(self.system, self.alignment_points)

    def run_base_alignment(self):
        # location of the alignment points in camera coordinates
        alignment_point_cam_locs = get_point_locations(
            self.system, self.alignment_points
        )
        point_name_list = self.alignment_settings["base_align_point_names"]
        alignment_point_ant_locs = [key_features_ant[i] for i in point_name_list]
        A_cam_to_ant, ant_scale, _ = procrustes_analysis(
            alignment_point_cam_locs, alignment_point_ant_locs
        )

        return A_cam_to_ant, ant_scale

    def move_points_to_mesh(self, A_cam_ant, ant_scale, camera_points):
        A_cam_ant = A_cam_ant.copy()
        A_cam_ant[:3, 3] *= ant_scale
        camera_points = np.asarray(camera_points) * ant_scale

        ant_points = matmul(A_cam_ant, camera_points)
        ant_points[:, 1] *= -1  # mirror y points
        mesh_points = matmul(np.linalg.inv(self.M_mesh_ant), ant_points)
        return mesh_points * self.mesh_scale

    # different arguments for easier minimization
    def _move_points_to_mesh(self, vals, camera_points, from_inverse=False):
        x, y, z, roll, pitch, yaw, scale = vals
        M = matrix_from_rot_trans(x, y, z, roll, pitch, yaw)
        if from_inverse:
            M = np.linalg.inv(M)

        return self.move_points_to_mesh(M, scale, camera_points)

    # key_feature_pixels should be 4x2 array
    # containing pixel locations where the key features are projected to
    # with some initial value for A and s.
    # deviation from these points can be penalized.
    # maybe just in a linear way for now
    def _minimization_function(
        self, vals, camera_points, tree, init_vals=None  # , key_feature_pixels=None
    ):
        mesh_points = self._move_points_to_mesh(
            vals, camera_points=camera_points, from_inverse=True
        )

        # penalize deviation from initial value
        if init_vals is not None:
            # should have shape of (7)
            val_diff = np.abs(np.asarray(vals) - np.asarray(init_vals))
            weights = self.alignment_settings["deviation_weights"]
            dev_loss = val_diff**2 * weights
            dev_loss = np.mean(dev_loss)
        else:
            dev_loss = 0

        distances, _ = tree.query(mesh_points)
        base_weight = self.alignment_settings["base_loss_weight"]
        huber_delta = self.alignment_settings["huber_delta"]
        base_loss = scipy.special.huber(huber_delta, distances)
        base_loss = np.mean(base_loss)
        base_loss = base_loss * base_weight
        loss = base_loss + dev_loss
        return loss

    def refine_matrix(
        self, A_cam_ant_init, ant_scale_init, camera_points, change_scale=True
    ):
        sample_vertices, _ = trimesh.sample.sample_surface(
            self.mesh,
            len(self.mesh.vertices) * self.alignment_settings["vertex_sample_density"],
        )
        tree = KDTree(sample_vertices)

        # we'll refine from the ant POV
        A_ant_cam_init = np.linalg.inv(A_cam_ant_init)
        x, y, z, roll, pitch, yaw = rot_trans_from_matrix(A_ant_cam_init)

        init_guess = np.asarray([x, y, z, roll, pitch, yaw, ant_scale_init])
        bounds = np.zeros((init_guess.shape[0], 2))
        bounds[:, 0] = init_guess + self.alignment_settings["refinement_bounds"]
        bounds[:, 1] = init_guess - self.alignment_settings["refinement_bounds"]

        if not change_scale:
            bounds[-1, 0] = init_guess[-1]
            bounds[-1, 1] = init_guess[-1]
        bounds = np.sort(bounds, axis=1)

        min_function = functools.partial(
            self._minimization_function,
            camera_points=camera_points,
            tree=tree,
            init_vals=init_guess,
        )

        res = minimize(fun=min_function, x0=init_guess, bounds=bounds)
        x2, y2, z2, roll2, pitch2, yaw2, scale2 = res.x

        A = matrix_from_rot_trans(x2, y2, z2, roll2, pitch2, yaw2)

        # switch back
        A = np.linalg.inv(A)
        return A, scale2
    
    def align_strike(
        self,
        strike_number,
        start_strike=None,
        match_sequentially=True,
        flow_parameters=None,
        reshift=True,
    ):
        if flow_parameters is None:
            flow_parameters = self.flow_settings
        run = True
        bad_numbers = []

        threshold = self.alignment_settings["point_error_cutoff"]

        if not match_sequentially and start_strike is None:
            start_strike = 1
        elif match_sequentially and start_strike is None:
            start_strike = max(1, strike_number - 1)

        if start_strike not in self.stored_match_points:
            self.align_strike(start_strike, match_sequentially=match_sequentially)

        strike_match_points = {}
        prev_match_points = self.stored_match_points[start_strike]
        prev_images = self.data_manager.get_start_images(strike_number=start_strike)
        new_images = self.data_manager.get_start_images(strike_number)

        for key, prev_image in prev_images.items():
            new_image = new_images[key]
            pmp = np.asarray(prev_match_points[key])[:, :2]

            if reshift:
                # do rough shift according to pre-saved numbers
                # this is (0, 0) for most strikes
                rough_shift_p = self.rough_interstrike_alignment[start_strike]
                rough_shift_n = self.rough_interstrike_alignment[strike_number]
                rough_shift = (rough_shift_n[0] - rough_shift_p[0],
                               rough_shift_n[1] - rough_shift_p[1])
                translation_matrix = np.float32([[1, 0, rough_shift[0]], [0, 1, rough_shift[1]]])
                prev_image = cv2.warpAffine(
                    prev_image.T, translation_matrix, prev_image.shape[:2]
                ).T
                pmp = pmp + rough_shift

                # this could get thrown off by consistent background or mandibles
                # so we'll crop in around the actual points
                buffer = 15 
                minx = int(np.min(pmp[:, 0]) - buffer) 
                maxx = int(np.max(pmp[:, 0]) + buffer) 
                miny = int(np.min(pmp[:, 1]) - buffer) 
                maxy = int(np.max(pmp[:, 1]) + buffer)

                minx = max(minx, 0)
                maxx = min(maxx, prev_image.shape[0]) 
                miny = max(miny, 0)
                maxy = min(maxy, prev_image.shape[1])

                i0 = prev_image[minx:maxx, miny:maxy].copy()
                i1 = new_image[minx:maxx, miny:maxy].copy()
                shiftx, shifty = cross_image(i1, i0)

                translation_matrix = np.float32([[1, 0, shiftx], [0, 1, shifty]])
                prev_image = cv2.warpAffine(
                    prev_image.T, translation_matrix, prev_image.shape[:2]
                ).T
                pmp = pmp + [shiftx, shifty]

            new_points = match_points_between_images(
                prev_image,
                new_image,
                pmp,
                flow_parameters=flow_parameters,
            )

            strike_match_points[key] = new_points

        # convert to camera points
        strike_locations = get_point_locations(self.system, strike_match_points)
        prev_strike_locations = get_point_locations(self.system, prev_match_points)

        strike_point_indices = np.arange(len(prev_match_points[2]))

        while run:
            # fit a transformation between the two
            A_cam2_to_cam1, _, transformed_points = procrustes_analysis(
                strike_locations[strike_point_indices],
                prev_strike_locations[strike_point_indices],
                allow_scale=False,
            )

            # look at error
            diff = np.linalg.norm(
                prev_strike_locations[strike_point_indices] - transformed_points,
                axis=1,
            )
            bp = np.where(diff > threshold)[0]

            if len(bp) > 0:
                worst_index = np.argsort(diff)[-1]
                bad_numbers.append(int(strike_point_indices[worst_index]))
                strike_point_indices = np.delete(strike_point_indices, worst_index)
            else:
                run = False

        for key, item in strike_match_points.items():
            strike_match_points[key] = np.asarray(item)[strike_point_indices]

        self.stored_match_points[strike_number] = strike_match_points
        prev_point_numbers = self.stored_point_numbers[start_strike]

        point_numbers = prev_point_numbers[strike_point_indices]
        self.stored_point_numbers[strike_number] = point_numbers

        return A_cam2_to_cam1, strike_match_points, point_numbers, bad_numbers, start_strike
    
    # def align_strike(
    #     self,
    #     strike_number,
    #     start_strike=None,
    #     match_sequentially=True,
    #     flow_parameters=None,
    #     reshift=True,
    # ):
    #     
    #     print("2024/11/21 you should do a rough alignment with larger window size, then refine with smaller window size")
    #     run = True
    #     bad_numbers = []

    #     threshold = self.alignment_settings["point_error_cutoff"]

    #     if not match_sequentially and start_strike is None:
    #         start_strike = 1
    #     elif match_sequentially and start_strike is None:
    #         start_strike = max(1, strike_number - 1)


    #     if start_strike not in self.stored_match_points:
    #         self.align_strike(start_strike, match_sequentially=match_sequentially)


    #     strike_match_points = {}
    #     prev_match_points = self.stored_match_points[start_strike]
    #     prev_images = self.data_manager.get_start_images(strike_number=start_strike)
    #     new_images = self.data_manager.get_start_images(strike_number)


    #     for key, prev_image in prev_images.items():
    #         new_image = new_images[key]
    #         pmp = np.asarray(prev_match_points[key])[:, :2]


    #         new_points = match_points_between_images(
    #             prev_image,
    #             new_image,
    #             pmp,
    #             flow_parameters=flow_parameters,
    #         )

    #         if reshift:
    #             # shift the previous image and start again
    #             shift = new_points - pmp
    #             # print(shift)
    #             shifty = np.median(shift[:, 1])
    #             shiftx = np.median(shift[:, 0])
    #             # print("first try")
    #             # print(shiftx, shifty)
    #             translation_matrix = np.float32([[1, 0, shifty], [0, 1, shiftx]])
    #             prev_image_shifted = cv2.warpAffine(
    #                 prev_image.T, translation_matrix, prev_image.shape[:2]
    #             ).T
    #             pmp_shifted = pmp + [shiftx, shifty]

    #             new_points = match_points_between_images(
    #                 prev_image_shifted,
    #                 new_image,
    #                 pmp_shifted,
    #                 flow_parameters=flow_parameters,
    #             )

    #             # # can get rid of this part
    #             # shift = new_points - pmp_shifted
    #             # #print(shift)
    #             # shiftx = np.median(shift[:, 0])
    #             # shifty = np.median(shift[:, 1])
    #             # print("second try")
    #             # print(shiftx, shifty)

    #         strike_match_points[key] = new_points


    #     # convert to camera points
    #     strike_locations = get_point_locations(self.system, strike_match_points)
    #     prev_strike_locations = get_point_locations(self.system, prev_match_points)

    #     strike_point_indices = np.arange(len(prev_match_points[2]))


    #     while run:
    #         # fit a transformation between the two
    #         A_cam2_to_cam1, _, transformed_points = procrustes_analysis(
    #             strike_locations[strike_point_indices],
    #             prev_strike_locations[strike_point_indices],
    #             allow_scale=False,
    #         )


    #         # look at error
    #         diff = np.linalg.norm(
    #             prev_strike_locations[strike_point_indices] - transformed_points,
    #             axis=1,
    #         )


    #         bp = np.where(diff > threshold)[0]

    #         if len(bp) > 0:
    #             worst_index = np.argsort(diff)[-1]
    #             bad_numbers.append(int(strike_point_indices[worst_index]))
    #             strike_point_indices = np.delete(strike_point_indices, worst_index)
    #         else:
    #             run = False


    #     for key, item in strike_match_points.items():
    #         strike_match_points[key] = np.asarray(item)[strike_point_indices]


    #     self.stored_match_points[strike_number] = strike_match_points
    #     prev_point_numbers = self.stored_point_numbers[start_strike]

    #     point_numbers = prev_point_numbers[strike_point_indices]
    #     self.stored_point_numbers[strike_number] = point_numbers


    #     return A_cam2_to_cam1, strike_match_points, point_numbers, bad_numbers, start_strike

    # def align_strike(self, strike_number):
    #     run = True
    #     bad_numbers = []
    #     point_numbers = np.arange(len(self.match_points[0]))
    #     threshold = self.alignment_settings["point_error_cutoff"]

    #     strike_match_points = {}
    #     prev_images = self.data_manager.get_start_images(strike_number=1)
    #     new_images = self.data_manager.get_start_images(strike_number)
    #     for key, prev_image in prev_images.items():
    #         new_image = new_images[key]
    #         new_points = match_points_between_images(
    #             prev_image, new_image, np.asarray(self.match_points[key])[:, :2]
    #         )
    #         strike_match_points[key] = new_points

    #     # convert to camera points
    #     strike_locations = get_point_locations(self.system, strike_match_points)

    #     while run:
    #         # fit a transformation between the two
    #         A_cam2_to_cam1, _, transformed_points = procrustes_analysis(
    #             strike_locations[point_numbers],
    #             self.point_camera_locations[point_numbers],
    #             allow_scale=False,
    #         )

    #         # look at error
    #         diff = np.linalg.norm(
    #             self.point_camera_locations[point_numbers] - transformed_points, axis=1
    #         )
    #         bp = np.where(diff > threshold)[0]
    #         if len(bp) > 0:
    #             worst_index = np.argsort(diff)[-1]
    #             bad_numbers.append(int(point_numbers[worst_index]))
    #             point_numbers = np.delete(point_numbers, worst_index)
    #         else:
    #             run = False

    #     for key, item in strike_match_points.items():
    #         strike_match_points[key] = np.asarray(item)[point_numbers]

    #     return A_cam2_to_cam1, strike_match_points, point_numbers, bad_numbers

    def run_strike1_alignment(self):
        if self.base_alignment_values is None:
            A_init, scale_init = self.run_base_alignment()
        else:
            bav = np.asanyarray(self.base_alignment_values)
            scale_init = bav[-1]
            x, y, z, roll, pitch, yaw = bav[:-1]
            A_ = matrix_from_rot_trans(x, y, z, roll, pitch, yaw)
            A_init = np.linalg.inv(A_)

        A_cam_ant, scale = self.refine_matrix(
            A_init,
            scale_init,
            self.point_camera_locations,
            change_scale=not self.alignment_settings["use_init_scale"],
        )
        self.A_cam_ant = A_cam_ant
        self.ant_scale = scale

        self.stored_alignment_matrices[1] = self.A_cam_ant

        return A_cam_ant, scale

    def determine_stable_points(self):
        if self.A_cam_ant is None or self.ant_scale is None:
            self.run_strike1_alignment()

        ant_points = matmul(self.A_cam_ant, self.point_camera_locations)
        hull = Polygon(stable_vertices_ant / self.ant_scale)
        stable = np.asarray(
            [hull.contains(Point(p[:2])) for p in ant_points], dtype=bool
        )

        self.stable_points = stable
        return stable

    def prepare_strike_results(self, strike_number, start_strike=None):
        if self.A_cam_ant is None or self.ant_scale is None:
            self.run_strike1_alignment()
        if self.stable_points is None:
            self.determine_stable_points()

        A_cam2_to_cam1, strike_match_points, point_numbers, bad_numbers, start_strike = (
            self.align_strike(strike_number, 
                              match_sequentially=self.alignment_settings["point_match_sequentially"],
                              reshift=self.alignment_settings["reshift_during_inter_strike_point_matching"],
                              start_strike=start_strike)
        )
        
        A_cam_ant = self.stored_alignment_matrices[start_strike]
        strike_A_cam_ant = np.linalg.matmul(A_cam_ant, A_cam2_to_cam1)

        # 2024/11/26
        # this should be cleaned up. If strikes aren't prepared in order
        # this step will fail
        self.stored_alignment_matrices[strike_number] = strike_A_cam_ant

        result_dict = {
            "mesh_filename": self.mesh_filename,
            "mesh_scale": self.mesh_scale,
            "M_mesh_ant": self.M_mesh_ant,
            "strike1_match_points": self.match_points,
            "match_points": strike_match_points,
            "alignment_points": self.alignment_points,
            "key_feature_coordinates": key_features_ant,
            "alignment_settings": self.alignment_settings,
            "A_cam_to_ant_start": strike_A_cam_ant,
            "A_cam_to_ant_start_strike1": self.A_cam_ant,
            "stable_points": self.stable_points[point_numbers],
            "point_numbers": point_numbers,
            "removed_points": bad_numbers,
            "ant_scale": self.ant_scale,
            "specimen_number": self.specimen_name,
            "strike_number": strike_number,
            "base_alignment_values": self.base_alignment_values,
            "aligned_from_strike_number": start_strike,
            "rough_interstrike_alignment": self.rough_interstrike_alignment,
            "strike_alignment_flow_settings": self.flow_settings
        }

        for key in save_keys:
            assert key in result_dict

        return result_dict


if __name__ == "__main__":
    a = Aligner("20240502_OB_3")
    example_results = a.prepare_strike_results(strike_number=3)
