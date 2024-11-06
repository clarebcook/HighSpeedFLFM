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
)

from .processing_functions import get_point_locations, match_points_between_images

import trimesh
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
        [0.8, 1.3],  # x
        [0.8, 1.3],  # y
        [0.9, 1.2],  # z - limited more to prevent points going to the center
        [0.8, 1.3],  # roll
        [0.8, 1.3],  # pitch
        [0.8, 1.3],  # yaw
        [0.8, 1.3],  # scale
    ],
    "scale_weight": 1e11,
    "huber_delta": 4000,
    "point_error_cutoff": 0.025,  # mm
    "use_init_scale": True,
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
    "strike_number"
]


class Aligner:
    def __init__(self, specimen_name, alignment_settings={}):
        self.mesh_filename = mesh_filename
        self.mesh_scale = mesh_scale
        self.M_mesh_ant = M_mesh_ant
        self.mesh = trimesh.load(home_directory + "/" + self.mesh_filename)

        self.specimen_name = specimen_name

        self.data_manager = MetadataManager(specimen_number=specimen_name)

        self.match_points = load_dictionary(self.data_manager.match_points_filename)
        self.alignment_points = load_dictionary(
            self.data_manager.alignment_points_filename
        )

        self.system = FLF_System(self.data_manager.calibration_filename)

        self.alignment_settings = default_alignment_settings
        for key, value in alignment_settings.items():
            self.alignment_settings[key] = value

        self.stable_points = None
        self.A_cam_ant = None
        self.ant_scale = None

    @property
    def point_camera_locations(self):
        return get_point_locations(self.system, self.match_points)

    def run_base_alignment(self):
        # location of the alignment points in camera coordinates
        alignment_point_cam_locs = get_point_locations(
            self.system, self.alignment_points
        )
        point_name_list = ["head_base", "eye_tip", "under_eye_ridge", "ridge_top"]
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
    def _move_points_to_mesh(self, vals, camera_points):
        x, y, z, roll, pitch, yaw, scale = vals
        M = matrix_from_rot_trans(x, y, z, roll, pitch, yaw)
        return self.move_points_to_mesh(M, scale, camera_points)

    def _minimization_function(self, vals, camera_points, tree):
        mesh_points = self._move_points_to_mesh(vals, camera_points=camera_points)
        scale = vals[-1]

        distances, _ = tree.query(mesh_points)
        weight = self.alignment_settings["scale_weight"]
        huber_delta = self.alignment_settings["huber_delta"]
        loss = np.mean(scipy.special.huber(huber_delta, distances)) + weight * 1 / scale
        return loss

    def refine_matrix(
        self, A_cam_ant_init, ant_scale_init, camera_points, change_scale=True
    ):
        sample_vertices, _ = trimesh.sample.sample_surface(
            self.mesh,
            len(self.mesh.vertices) * self.alignment_settings["vertex_sample_density"],
        )
        tree = KDTree(sample_vertices)

        x, y, z, roll, pitch, yaw = rot_trans_from_matrix(A_cam_ant_init)

        init_guess = np.asarray([x, y, z, roll, pitch, yaw, ant_scale_init])
        bounds = np.asarray(self.alignment_settings["refinement_bounds"])
        bounds[:, 0] *= init_guess
        bounds[:, 1] *= init_guess

        if not change_scale:
            bounds[-1, 0] = init_guess[-1]
            bounds[-1, 1] = init_guess[-1]
        bounds = np.sort(bounds, axis=1)

        min_function = functools.partial(
            self._minimization_function, camera_points=camera_points, tree=tree
        )

        res = minimize(fun=min_function, x0=init_guess, bounds=bounds)
        x2, y2, z2, roll2, pitch2, yaw2, scale2 = res.x

        A = matrix_from_rot_trans(x2, y2, z2, roll2, pitch2, yaw2)
        return A, scale2

    def align_strike(self, strike_number):
        run = True
        bad_numbers = []
        point_numbers = np.arange(len(self.match_points[0]))
        threshold = self.alignment_settings["point_error_cutoff"]

        strike_match_points = {}
        prev_images = self.data_manager.dark_calibration_images
        new_images = self.data_manager.get_start_images(strike_number)
        for key, prev_image in prev_images.items():
            new_image = new_images[key]
            new_points = match_points_between_images(
                prev_image, new_image, np.asarray(self.match_points[key])
            )
            strike_match_points[key] = new_points

        # convert to camera points
        strike_locations = get_point_locations(self.system, strike_match_points)

        while run:
            # fit a transformation between the two
            A_cam2_to_cam1, _, transformed_points = procrustes_analysis(
                strike_locations[point_numbers],
                self.point_camera_locations[point_numbers],
                allow_scale=False,
            )

            # look at error
            diff = np.linalg.norm(
                self.point_camera_locations[point_numbers] - transformed_points, axis=1
            )
            bp = np.where(diff > threshold)[0]
            if len(bp) > 0:
                worst_index = np.argsort(diff)[-1]
                bad_numbers.append(int(point_numbers[worst_index]))
                point_numbers = np.delete(point_numbers, worst_index)
            else:
                run = False

        for key, item in strike_match_points.items():
            strike_match_points[key] = np.asarray(item)[point_numbers]

        return A_cam2_to_cam1, strike_match_points, point_numbers, bad_numbers

    def run_strike1_alignment(self):
        A_init, scale_init = self.run_base_alignment()
        A_cam_ant, scale = self.refine_matrix(
            A_init,
            scale_init,
            self.point_camera_locations,
            change_scale=not self.alignment_settings["use_init_scale"],
        )
        self.A_cam_ant = A_cam_ant
        self.ant_scale = scale

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

    def prepare_strike_results(self, strike_number):
        if self.A_cam_ant is None or self.ant_scale is None:
            self.run_strike1_alignment()
        if self.stable_points is None:
            self.determine_stable_points()

        A_cam2_to_cam1, strike_match_points, point_numbers, bad_numbers = (
            self.align_strike(strike_number)
        )

        strike_A_cam_ant = np.linalg.matmul(self.A_cam_ant, A_cam2_to_cam1)

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
            "strike_number": strike_number
        }

        for key in save_keys:
            assert key in result_dict

        return result_dict


if __name__ == "__main__":
    a = Aligner("20240502_OB_3")
    example_results = a.prepare_strike_results(strike_number=3)
