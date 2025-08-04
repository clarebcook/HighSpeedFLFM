from hsflfm.util import load_dictionary, matmul, MetadataManager, save_dictionary
from hsflfm.ant_model import M_mesh_ant, mesh_scale
from .basic_functions import (
    get_peak_indices,
    get_strike_center,
    sort_by_camera,
    convert_to_percentile,
)

from tqdm import tqdm
import torch
import numpy as np
from scipy.spatial import cKDTree
import os
from pathlib import Path

load_value_list = [
    "specimen_number",
    "strike_number",
    "point_number",
    "peak_indices",
    "displacement",
    "normalized_displacement",
    "start_locations_mm",
    "start_locations_std",
    "flow_error_at_peak",
    "average_flow_error",
    "flow_at_peak_sq",
    "average_flow_sq",
    "mesh_points",
    # "huber_loss_at_peak",
    # "average_huber_loss",
    "strike_center_indices",
    "mandible_start_frames",
    "mandible_order",
    "max_z_velocity",
]


class ResultManager:
    def __init__(self, result_info):
        self.result_info = result_info

    @property
    def rel_displacements(self):
        return torch.asarray(self.result_info["rel_displacements"])

    def max_abs_velocity(self, dim=2, frame_restriction=5, return_indices=False):
        displacements = self.rel_displacements[:, :, dim]
        derivs = torch.diff(displacements, axis=1)

        if frame_restriction is not None:
            center_index = self.strike_center_index(dim=dim)
            si = max(0, center_index - frame_restriction)
            ei = min(center_index + frame_restriction, derivs.shape[1])
            derivs = derivs[:, si:ei]

        abs_derivs = torch.abs(derivs)
        _, indices = torch.max(abs_derivs, axis=1)
        max_vals = derivs[torch.arange(derivs.shape[0]), indices]

        if return_indices:
            # there's definitely a cleaner way to do this
            if frame_restriction is not None:
                indices = indices + si
            return max_vals, indices

        return max_vals

    def peak_indices(self, dim=2):
        arr = self.rel_displacements
        return get_peak_indices(arr[:, :, dim]).to(torch.int)

    def peak_displacements(self, dim=2):
        displacements = self.rel_displacements[:, :, dim]
        p = torch.arange(displacements.shape[0])
        return displacements[p, self.peak_indices()]

    def peak_norm_displacements(self, dim=2):
        peak_disp = self.peak_displacements(dim=dim)
        peak_disp = peak_disp / torch.mean(torch.abs(peak_disp))
        return peak_disp

    def strike_center_index(self, dim=2):
        disp = self.rel_displacements[:, :, dim]
        return get_strike_center(disp)

    @property
    def point_start_locs_camera(self):
        return torch.asarray(self.result_info["camera_start_locations"])

    @property
    def point_start_locs_ant_mm(self):
        start_locs = self.point_start_locs_camera
        ant_locs = matmul(
            torch.asarray(self.result_info["A_cam_to_ant_start"]), start_locs
        )
        return ant_locs

    @property
    def point_start_locs_ant_std(self):
        start_locs = self.point_start_locs_camera * self.result_info["ant_scale"]
        matrix = torch.asarray(self.result_info["A_cam_to_ant_start"])
        matrix[:3, 3] *= self.result_info["ant_scale"]

        locations = matmul(matrix, start_locs)
        return locations

    @property
    def point_mesh_locations(self):
        ant_locs = self.point_start_locs_ant_std
        ant_locs[:, 1] *= -1
        matrix = torch.asarray(M_mesh_ant, dtype=ant_locs.dtype)
        matrix = torch.linalg.inv(matrix)
        mesh_points = matmul(matrix, ant_locs)
        return mesh_points * mesh_scale

    @property
    def flow_differences(self):
        predictions = torch.asarray(self.result_info["predicted_flow_vectors"])
        flow_vectors = torch.asarray(self.result_info["flow_vectors"])
        diff = flow_vectors - predictions
        return diff

    # def peak_huber_loss(self, dim=2):
    #     peak_indices = self.peak_indices(dim=dim)
    #     loss = torch.asarray(self.result_info["huber_loss"])
    #     p = torch.arange(loss.shape[0])
    #     peak_loss = loss[p, :, peak_indices]
    #     return peak_loss

    # def huber_loss_around_strike(self, center_index=None, half_length=12):
    #     if center_index is None:
    #         center_index = self.strike_center_index(dim=2)

    #     loss = torch.asarray(self.result_info["huber_loss"])
    #     # avergae losses over a range
    #     loss_c = loss[
    #         :,
    #         :,
    #         max(center_index - half_length, 0) : min(
    #             center_index + half_length + 1, loss.shape[-1]
    #         ),
    #     ]

    #     return torch.mean(torch.abs(loss_c), axis=-1)

    def peak_flow_differences(self, dim=2):
        peak_indices = self.peak_indices(dim=dim)
        diff = self.flow_differences
        p = torch.arange(diff.shape[0])
        peak_diff = diff[p, :, peak_indices]
        return peak_diff

    def flow_diff_around_strike(self, center_index=None, half_length=12, square=False):
        if center_index is None:
            center_index = self.strike_center_index(dim=2)

        diff = self.flow_differences
        # avergae differences over a range
        diff_c = diff[
            :,
            :,
            max(center_index - half_length, 0) : min(
                center_index + half_length + 1, diff.shape[-1]
            ),
        ]
        if square:
            diff_c = diff_c**2
        return torch.mean(torch.abs(diff_c), axis=-1)

    # this is what we're currently using as the main error score
    # average flow error squared from the top two cameras
    @property
    def error_scores(self):
        average_flow_sq = self.flow_diff_around_strike(half_length=12, square=True)
        _, sorted = sort_by_camera(average_flow_sq[:, :, None], treat_individually=True)
        return torch.mean(sorted[:, :2].squeeze(), axis=1)


# 20250804
# this could likely be done in a cleaner way
# but is currently the easiest method, though it makes some assumptions
# about file organization
def build_bulk_analyzer(result_folder, save_results=True, reload=False):
    res_filename = f"{result_folder}/loaded_results.json"
    if os.path.exists(res_filename) and not reload:
        analyzer = BulkAnalyzer(None)
        analyzer.all_results = load_dictionary(res_filename)
        for key, value in analyzer.all_results.items():
            if key in ["specimen_number", "mandible_order"]:
                analyzer.all_results[key] = np.asarray(value)
                continue
            analyzer.all_results[key] = torch.asarray(value)
    else:
        # get names of all filenames
        folders = os.listdir(result_folder)
        all_filenames = []
        for inner in folders:
            path = Path(result_folder) / inner
            if path.is_dir():
                filenames = os.listdir(path)
                for filename in filenames:
                    if filename[-4:] == "json":
                        all_filenames.append(str(path / filename))
        analyzer = BulkAnalyzer(all_filenames)
        analyzer.load_results()
        if save_results:
            save_dictionary(analyzer.all_results, res_filename)
    return analyzer


class BulkAnalyzer:
    def __init__(self, all_result_filenames):
        self.all_result_filenames = all_result_filenames
        return

    def load_results(self):
        loaded_results = None

        for filename in tqdm(self.all_result_filenames):
            result_dict = load_dictionary(filename)
            result_manager = ResultManager(result_dict)

            rel_displacements = result_manager.rel_displacements
            peak_displacements = torch.empty(
                (rel_displacements.shape[0], rel_displacements.shape[2]),
                dtype=rel_displacements.dtype,
            )
            peak_norm_displacements = torch.zeros_like(peak_displacements)
            peak_indices = torch.zeros(peak_displacements.shape, dtype=torch.int)
            max_velocities = torch.zeros_like(peak_displacements)
            for dim in range(3):
                peak_indices[:, dim] = result_manager.peak_indices(dim=dim)
                peak_displacements[:, dim] = result_manager.peak_displacements(dim=dim)
                peak_norm_displacements[:, dim] = (
                    result_manager.peak_norm_displacements(dim=dim)
                )
                max_velocities[:, dim] = result_manager.max_abs_velocity(dim=dim)

            start_locations_ant_mm = result_manager.point_start_locs_ant_mm
            start_locations_ant_std = result_manager.point_start_locs_ant_std
            start_locations_mesh = result_manager.point_mesh_locations

            peak_flow_diff = result_manager.peak_flow_differences()
            range_flow_diff = result_manager.flow_diff_around_strike(half_length=12)

            peak_flow_sq = result_manager.peak_flow_differences() ** 2
            average_flow_sq = result_manager.flow_diff_around_strike(
                half_length=12, square=True
            )

            center_index = result_manager.strike_center_index()

            # peak_huber_loss = result_manager.peak_huber_loss()
            # range_huber_loss = result_manager.huber_loss_around_strike(half_length=12)

            num_points = len(result_dict["point_numbers"])

            # get mandible information
            metadata_manager = MetadataManager(result_dict["specimen_number"])
            start_frames = metadata_manager.mandible_start_frames(
                result_dict["strike_number"]
            )
            start_frames = torch.asarray([start_frames] * num_points)
            mandible_order = metadata_manager.mandible_order(
                result_dict["strike_number"]
            )
            mandible_order = np.asarray([mandible_order] * num_points)

            array_list = [
                np.asarray([result_dict["specimen_number"]] * num_points),
                torch.asarray([result_dict["strike_number"]] * num_points),
                torch.asarray(result_dict["point_numbers"]),
                peak_indices,
                peak_displacements,
                peak_norm_displacements,
                start_locations_ant_mm,
                start_locations_ant_std,
                peak_flow_diff,
                range_flow_diff,
                peak_flow_sq,
                average_flow_sq,
                start_locations_mesh,
                # peak_huber_loss,
                # range_huber_loss,
                torch.asarray([center_index] * num_points),
                start_frames,
                mandible_order,
                max_velocities,
            ]

            if loaded_results is None:
                loaded_results = {}
                for key, value in zip(load_value_list, array_list):
                    loaded_results[key] = value
            else:
                for key, value in zip(load_value_list, array_list):
                    if key in ["specimen_number", "mandible_order"]:
                        loaded_results[key] = np.concatenate(
                            (loaded_results[key], value), axis=0
                        )
                        continue
                    loaded_results[key] = torch.concatenate(
                        (loaded_results[key], value), axis=0
                    )

        self.all_results = loaded_results

    # this should drop all points above the error threshold
    # from the dictionary
    def enforce_error_threshold(self, error_threshold=0.0015):
        error_scores = self.error_scores
        good_indices = torch.where(error_scores < error_threshold)[0]
        for key, item in self.all_results.items():
            self.all_results[key] = item[good_indices]

    # this is a little hand-wavy
    # but it's been a useful way to compare spatial trends
    # for strikes with different strength
    def compute_strength_scores(self, error_threshold=0.0015, k=15):
        all_error_scores = self.error_scores
        good_indices = torch.where(all_error_scores < error_threshold)
        _, neighbor_indices = self.get_closest_point_indices(k=k, indices=good_indices)

        strength_scores = torch.zeros_like(self.error_scores)
        named_scores = {}

        # loop through the represented strikes
        specimens = np.unique(self.all_results["specimen_number"])
        for specimen in specimens:
            named_scores[specimen] = {}
            specimen_indices = self.get_specimen_indices(specimen)
            strikes = np.unique(self.all_results["strike_number"][specimen_indices])
            for strike in strikes:
                strike_all_indices = self.get_specimen_indices(specimen, strike)
                strike_indices = np.intersect1d(strike_all_indices, good_indices)

                strike_neighbor_indices = neighbor_indices[strike_indices]

                ratios = np.zeros(strike_neighbor_indices.shape[0])
                for pi, neighbor_index in enumerate(strike_neighbor_indices):
                    displacements = self.all_results["displacement"][neighbor_index]
                    disp_norm = np.linalg.norm(displacements, axis=-1)
                    point_disp = np.linalg.norm(
                        self.all_results["displacement"][strike_indices[pi]]
                    )
                    ratios[pi] = point_disp / np.mean(disp_norm)
                score = np.mean(ratios)

                named_scores[specimen][strike] = score

                # assign the score
                strength_scores[strike_all_indices] = score

        self.all_results["strength_score"] = strength_scores

        return named_scores

    def get_top_values(self, array_type, num_cams=2, treat_individually=True):
        okay_keys = [
            "flow_error_at_peak",
            "average_flow_error",
            # "huber_loss_at_peak",
            # "average_huber_loss",
            "flow_at_peak_sq",
            "average_flow_sq",
        ]
        if array_type not in okay_keys:
            raise ValueError(
                f"""{array_type} not available,
                              must be one of {okay_keys}"""
            )
        array = torch.abs(self.all_results[array_type])
        _, sorted = sort_by_camera(
            array[:, :, None], treat_individually=treat_individually
        )
        return sorted[:, :num_cams].squeeze()

    def get_percentile(self, array_type, num_cams=3):
        arr = torch.mean(self.get_top_values(array_type, num_cams=num_cams), axis=1)

        arr_percent = convert_to_percentile(arr.numpy())
        return torch.asarray(arr_percent)

    # function that can take a location in the standardized ant coordinate system
    # and return the indices of the k points closest to that location
    # indices can be used to indicate that only some points should be used
    def get_closest_point_indices(self, k=25, indices=None):
        all_points = self.all_results["start_locations_std"]
        if indices is not None:
            points = all_points[indices]
        else:
            points = all_points
        tree = cKDTree(points)

        distances, indices = tree.query(all_points, k=k + 1)
        neighbor_indices = indices[:, 1:]
        neighbor_distances = distances[:, 1:]

        return neighbor_distances, neighbor_indices

    # points should be in the ant coordinate system
    # in the standardized units
    def get_points_within_radius(self, target_point, radius):
        points = self.all_results["start_locations_std"]
        tree = cKDTree(points)  # could also find all points within some radius
        indices = tree.query_ball_point(target_point, radius)
        return indices

    # return the indices corresponding to specific ants
    def get_specimen_indices(self, specimen_name, strike_number=None):
        spec_indices = np.where(self.all_results["specimen_number"] == specimen_name)[0]
        if strike_number is None:
            return spec_indices

        strike_indices = np.where(self.all_results["strike_number"] == strike_number)[0]
        return np.intersect1d(spec_indices, strike_indices)

    # this is what we're currently using as the main error score
    @property
    def error_scores(self):
        return torch.mean(self.get_top_values("average_flow_sq", num_cams=2), axis=1)
