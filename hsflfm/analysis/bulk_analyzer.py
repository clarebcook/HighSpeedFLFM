from hsflfm.util import load_dictionary, matmul
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
    "huber_loss_at_peak",
    "average_huber_loss",
]


class ResultManager:
    def __init__(self, result_info):
        self.result_info = result_info

    @property
    def rel_displacements(self):
        return torch.asarray(self.result_info["rel_displacements"])

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

    def peak_huber_loss(self, dim=2):
        peak_indices = self.peak_indices(dim=dim)
        loss = torch.asarray(self.result_info["huber_loss"])
        p = torch.arange(loss.shape[0])
        peak_loss = loss[p, :, peak_indices]
        return peak_loss

    def huber_loss_around_strike(self, center_index=None, half_length=12):
        if center_index is None:
            center_index = self.strike_center_index(dim=2)

        loss = torch.asarray(self.result_info["huber_loss"])
        # avergae losses over a range
        loss_c = loss[
            :,
            :,
            max(center_index - half_length, 0) : min(
                center_index + half_length + 1, loss.shape[-1]
            ),
        ]

        return torch.mean(torch.abs(loss_c), axis=-1)

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
            for dim in range(3):
                peak_indices[:, dim] = result_manager.peak_indices(dim=dim)
                peak_displacements[:, dim] = result_manager.peak_displacements(dim=dim)
                peak_norm_displacements[:, dim] = (
                    result_manager.peak_norm_displacements(dim=dim)
                )

            start_locations_ant_mm = result_manager.point_start_locs_ant_mm
            start_locations_ant_std = result_manager.point_start_locs_ant_std
            start_locations_mesh = result_manager.point_mesh_locations

            peak_flow_diff = result_manager.peak_flow_differences()
            range_flow_diff = result_manager.flow_diff_around_strike(half_length=12)

            peak_flow_sq = result_manager.peak_flow_differences()**2
            average_flow_sq = result_manager.flow_diff_around_strike(half_length=12, square=True)

            peak_huber_loss = result_manager.peak_huber_loss()
            range_huber_loss = result_manager.huber_loss_around_strike(half_length=12)

            num_points = len(result_dict["point_numbers"])
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
                peak_huber_loss,
                range_huber_loss,
            ]

            if loaded_results is None:
                loaded_results = {}
                for key, value in zip(load_value_list, array_list):
                    loaded_results[key] = value
            else:
                for key, value in zip(load_value_list, array_list):
                    if key == "specimen_number":
                        loaded_results[key] = np.concatenate(
                            (loaded_results[key], value), axis=0
                        )
                        continue
                    loaded_results[key] = torch.concatenate(
                        (loaded_results[key], value), axis=0
                    )

        self.all_results = loaded_results

    def get_top_values(self, array_type, num_cams=2, treat_individually=True):
        okay_keys = [
            "flow_error_at_peak",
            "average_flow_error",
            "huber_loss_at_peak",
            "average_huber_loss",
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
