# description

# imports
from .processing_functions import (
    get_point_locations,
    get_flow_vectors,
    gaussian_filter,
    get_global_movement,
    estimate_affine_transform,
    solve_weighted_system,
)
from hsflfm.util import MetadataManager, load_split_video, get_timestamp
from hsflfm.calibration import FLF_System
from hsflfm.analysis import get_strike_center, sort_by_camera

import numpy as np
import torch
from scipy.ndimage import gaussian_filter1d

# default settings
default_regression_settings = {
    "data_scale": 100,
    "max_iterations": 3000,
    "filter_sigma": 3,
    "filter_size": 10,
    "alpha": 0.8,
    "weight_update_scale": 2,
    "weight_change_thresh": 1e-4,
}

default_flow_settings = {
    "pyr_scale": 0.5,
    "levels": 1,
    "winsize": 11,
    "iterations": 5,
    "poly_n": 1,
    "poly_sigma": 0.2,
    "flags": 0,
    "filter_sigma": 1,
}

default_global_movement_settings = {
    "stable_ratio": [1, 1],
    "gm_loss_sq_threshold": 0.0015,
}

new_keys = [
    "affine_matrices",
    "rel_displacements",
    "camera_point_displacements",
    "camera_start_locations",
    "global_movement",
    "time",
    "flow_settings",
    "regression_settings",
    "global_movement_settings",
    "camera_weights",
    "points_used_in_gm",
    "predicted_flow_vectors",
    "unfiltered_flow_vectors",
    "flow_vectors",
    "iteration_losses",
    "threshold_loss",
    "error_metric",
]


class StrikeProcessor:
    def __init__(
        self,
        alignment_results,
        flow_settings={},
        regression_settings={},
        global_movement_settings={},
    ):

        self.result_info = alignment_results
        self.data_manager = MetadataManager(self.result_info["specimen_number"])
        self.system = FLF_System(self.data_manager.calibration_filename)
        self.start_positions = get_point_locations(
            self.system, alignment_results["match_points"]
        )

        self.num_points = self.start_positions.shape[0]

        self.regression_settings = regression_settings
        for key, item in default_regression_settings.items():
            if key not in self.regression_settings:
                self.regression_settings[key] = item

        self.global_movement_settings = global_movement_settings
        for key, item in default_global_movement_settings.items():
            if key not in self.global_movement_settings:
                self.global_movement_settings[key] = item

        self.flow_settings = flow_settings
        for key, item in default_flow_settings.items():
            if key not in self.flow_settings:
                self.flow_settings[key] = item

        # these will fill in later
        self.all_iteration_losses = None
        self.all_weights = None
        self.all_displacements = None
        self.all_full_loss = None
        self.all_flow_predictions = None

        # pull out a few things for convenience
        self.match_points = self.result_info["match_points"]
        self.num_cameras = len(self.match_points)

        self.pixel_size_mm = self.system.calib_manager.pixel_size * 1e3

    def get_flow_vectors(self):
        settings = self.flow_settings.copy()
        filter_sigma = settings.pop("filter_sigma")

        video_filename = self.data_manager.video_filename(
            self.result_info["strike_number"]
        )
        videos = load_split_video(
            video_filename, self.data_manager.calibration_filename
        )
        self.num_frames = len(videos[0])
        for key, item in videos.items():
            videos[key] = np.mean(item, axis=-1)
        unfiltered_flow_vectors = get_flow_vectors(videos, self.match_points, settings)
        self.result_info["unfiltered_flow_vectors"] = unfiltered_flow_vectors

        # then filter the vectors as necessary
        if filter_sigma is not None:
            ufv = unfiltered_flow_vectors
            ufv_r = ufv.reshape((ufv.shape[0] * ufv.shape[1], ufv.shape[2]))
            for i, arr in enumerate(ufv_r):
                filtered = gaussian_filter1d(arr, sigma=filter_sigma)
                ufv_r[i] = filtered - filtered[0]
            flow_vectors = ufv_r.reshape(ufv.shape)
        else:
            flow_vectors = unfiltered_flow_vectors
        self.result_info["flow_vectors"] = flow_vectors
        self.flow_vectors = torch.asarray(flow_vectors)

    # get coefficients for the liner system of equations
    def _get_coeffs(self):
        all_coeffs = torch.zeros(self.num_points, self.num_cameras * 2, 3)

        # just fill in inefficiently for now
        for point_index in range(self.num_points):
            for cam_num in range(self.num_cameras):
                pixel_location = self.match_points[cam_num][point_index]
                z = self.start_positions[point_index][2]
                v0, v1 = self.system.get_shift_slopes(
                    cam_num, [pixel_location[0]], [pixel_location[1]]
                )

                mag0 = self.system.get_magnification_at_plane(cam_num, z, 0)
                mag1 = self.system.get_magnification_at_plane(cam_num, z, 1)

                all_coeffs[point_index, 2 * cam_num] = torch.asarray(
                    [mag0 / self.pixel_size_mm, 0, v0[0]]
                )
                all_coeffs[point_index, 2 * cam_num + 1] = torch.asarray(
                    [0, mag1 / self.pixel_size_mm, v1[0]]
                )

        return all_coeffs

    def run_regression(self, device="cpu"):
        filt = gaussian_filter(
            size=self.regression_settings["filter_size"],
            sigma=self.regression_settings["filter_sigma"],
        ).to(device)

        with torch.no_grad():
            dev = torch.device(device)
            data_scale = self.regression_settings["data_scale"]
            weight_change_thresh = self.regression_settings["weight_change_thresh"]
            max_iters = self.regression_settings["max_iterations"]
            weight_update_scale = self.regression_settings["weight_update_scale"]
            alpha = self.regression_settings["alpha"]

            measurements = self.flow_vectors.to(dev).to(torch.float32)
            weights = torch.ones_like(measurements)
            matrices = self._get_coeffs()

            all_weights = torch.zeros(
                (max_iters, weights.shape[0], weights.shape[1], weights.shape[2])
            )
            all_errors = torch.zeros_like(all_weights)

            for i in range(max_iters):
                values = solve_weighted_system(
                    matrices, weights, measurements * data_scale
                )
                flow_predictions = torch.matmul(matrices, values)
                errors = (flow_predictions - measurements * data_scale) ** 2
                camera_loss = errors.clone()

                # I'm sure there's a better way to do this
                # combining the flow losses into camera losses
                for cam_num in range(self.num_cameras):
                    cl = torch.mean(
                        camera_loss[:, 2 * cam_num : 2 * cam_num + 2], axis=1
                    )
                    camera_loss[:, [2 * cam_num]] = cl[:, None, :]
                    camera_loss[:, [2 * cam_num + 1]] = cl[:, None, :]
                loss_reshape = camera_loss.reshape(
                    (
                        camera_loss.shape[0] * camera_loss.shape[1],
                        1,
                        camera_loss.shape[2],
                    )
                )

                camera_loss = filt(loss_reshape).reshape(camera_loss.shape)
                weights = alpha * weights + (1 - alpha) * torch.exp(
                    -torch.abs(
                        weight_update_scale * torch.sqrt(camera_loss) / data_scale
                    )
                )

                all_weights[i] = weights
                all_errors[i] = errors
                # check if the weights have stopped updating
                if i == 0:
                    continue
                diff = torch.abs(all_weights[i - 1] - weights)
                if torch.max(diff) < weight_change_thresh:
                    break

            all_weights = all_weights[:i]
            all_errors = all_errors[:i]

        all_errors = torch.sqrt(all_errors) / data_scale
        self.all_displacements = values.permute(0, 2, 1).cpu().detach() / data_scale
        self.all_displacements[:, :, 2] *= -1
        self.all_iteration_losses = all_errors.cpu().detach()
        self.all_weights = weights[:, ::2].cpu().detach()
        self.all_flow_predictions = flow_predictions.cpu().detach() / data_scale

    def get_relative_displacements(self):
        if self.all_displacements is None:
            print("must run regression first!")
            return

        # Step 1: move points into ant coordinate system from first frame
        cam_locations = self.all_displacements + self.start_positions[:, None, :]
        cam_locations = torch.cat(
            (
                cam_locations,
                torch.ones((cam_locations.shape[0], cam_locations.shape[1], 1)),
            ),
            axis=-1,
        )

        cam_locations_flat = cam_locations.reshape(
            (self.num_frames * self.num_points, 4)
        )

        A_cam_to_ant = torch.asarray(self.result_info["A_cam_to_ant_start"]).to(
            cam_locations_flat.dtype
        )

        transformed_point_locs = torch.linalg.matmul(
            A_cam_to_ant, cam_locations_flat.T
        ).T
        transformed_point_locs = transformed_point_locs.reshape(cam_locations.shape)[
            :, :, :3
        ]

        # Step 2: determine which points have low enough loss to be used
        # we are using flow loss from the top two cameras
        loss_threshold = self.global_movement_settings["gm_loss_sq_threshold"]
        # this is just to compute the loss we want... could mvoe elsewhere
        flow_diff_sq = (self.all_flow_predictions - self.flow_vectors) ** 2
        strike_center = get_strike_center(self.all_displacements[:, :, 2])
        half_distance = 12
        start_frame = max(0, strike_center - half_distance)
        end_frame = max(flow_diff_sq.shape[-1], strike_center + half_distance)
        flow_diff_sq = flow_diff_sq[:, :, start_frame:end_frame]
        _, flow_diff_sq_sorted = sort_by_camera(flow_diff_sq, treat_individually=False)
        flow_diff_sq_top = flow_diff_sq_sorted[:, :2]
        flow_diff_sq_top = torch.mean(flow_diff_sq_top, axis=(1, 2))
        self.result_info["threshold_loss"] = flow_diff_sq_top
        self.low_loss_points = flow_diff_sq_top <= loss_threshold

        # if less than half the points are good, we adjust the threshold
        # to use half the points
        if np.count_nonzero(self.low_loss_points) < len(self.low_loss_points) / 2:
            self.global_movement_settings["gm_loss_sq_threshold"] = float(
                torch.median(self.result_info["threshold_loss"])
            )
            self.low_loss_points = flow_diff_sq_top <= loss_threshold

        self.error_metric = flow_diff_sq_top

        # Step 3: use stable ratio to give weighting to points
        stable_ratio = self.global_movement_settings["stable_ratio"]
        sp = torch.asarray(self.result_info["stable_points"]).to(bool)[
            torch.asarray(self.result_info["point_numbers"])
        ]
        sp = sp & self.low_loss_points
        nsp = ~sp & self.low_loss_points

        # Step 4: get global movement and affine matrices
        weighted_locations = torch.cat(
            (
                transformed_point_locs[sp].repeat(stable_ratio[0], 1, 1),
                transformed_point_locs[nsp],
            ),
            axis=0,
        )
        weighted_start_positions = weighted_locations[:, [0], :].repeat(
            1, weighted_locations.shape[1], 1
        )
        affine_matrices = estimate_affine_transform(
            original_points=weighted_start_positions,
            transformed_points=weighted_locations,
        ).to(transformed_point_locs.dtype)
        global_movement = get_global_movement(affine_matrices=affine_matrices)

        # Step 5: get the predicted point displacements
        start_positions = transformed_point_locs[:, 0]
        start_positions = torch.cat(
            (start_positions, torch.ones((self.num_points, 1))), axis=1
        )
        start_positions_batched = (
            start_positions[None].repeat(self.num_frames, 1, 1).permute(0, 2, 1)
        )
        predicted_displacements = torch.bmm(
            affine_matrices, start_positions_batched
        ).permute(0, 2, 1)[:, :, :3]

        # Step 6: get the relative displacements from first frame coordinate frame
        rel_displacements_first_frame = (
            transformed_point_locs.permute(1, 0, 2) - predicted_displacements
        )

        # Step 7: switch these to be the current coordinate frame
        rel_displacements = torch.bmm(
            affine_matrices[:, :3, :3], rel_displacements_first_frame.permute(0, 2, 1)
        )
        rel_displacements = rel_displacements.permute(2, 0, 1)

        self.rel_displacements = rel_displacements
        self.global_movement = global_movement
        self.affine_matrices = affine_matrices

    def condense_info(self):
        self.result_info["predicted_flow_vectors"] = self.all_flow_predictions.tolist()
        self.result_info["camera_weights"] = self.all_weights.tolist()
        self.result_info["regression_settings"] = self.regression_settings
        self.result_info["iteration_losses"] = self.all_iteration_losses.tolist()
        self.result_info["camera_point_displacements"] = self.all_displacements.tolist()
        self.result_info["points_used_in_gm"] = self.low_loss_points.tolist()
        self.result_info["rel_displacements"] = self.rel_displacements.tolist()
        self.result_info["affine_matrices"] = self.affine_matrices.tolist()
        self.result_info["camera_start_locations"] = self.start_positions.tolist()
        listed_global_movement = {}
        for key, item in self.global_movement.items():
            listed_global_movement[key] = item.tolist()
        self.result_info["global_movement"] = listed_global_movement

        self.result_info["regression_settings"] = self.regression_settings
        self.result_info["global_movement_settings"] = self.global_movement_settings
        self.result_info["flow_settings"] = self.flow_settings

        self.result_info["time"] = get_timestamp()
        self.result_info["error_metric"] = self.error_metric

        # all_keys = torch.cat((copy_keys, assert_keys, new_keys))
        for key in new_keys:
            assert key in self.result_info
        return self.result_info
