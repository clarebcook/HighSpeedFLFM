import xarray as xr
import numpy as np
import cv2

import torch
import torch.nn as nn
from torch.nn.functional import conv1d


# this is to bring us from a point in the world/camera frame of reference
# to a pixel in a camera
def world_frame_to_pixel(system, point, camera=2):
    ref_camera = system.reference_camera
    # use the magnification at that z-plane to switch x, y value back to pixels
    mag0 = system.get_magnification_at_plane(
        camera_number=ref_camera, plane_mm=point[2], dim=0
    )
    mag1 = system.get_magnification_at_plane(
        camera_number=ref_camera, plane_mm=point[2], dim=1
    )

    pixel_size_m = system.calib_manager.pixel_size
    pixels = (
        point[0] * mag0 / (pixel_size_m * 1e3),
        point[1] * mag1 / (pixel_size_m * 1e3),
    )

    # then shift to the other camera if it's not the reference camera
    s0, s1 = system.get_pixel_shifts(camera, [pixels[0]], [pixels[1]])
    pixels = (pixels[0] - s0, pixels[1] - s1)

    # then use height and calibration model to find the pixel
    # just using the reference camera to make things easier
    v0, v1 = system.get_shift_slopes(camera, [pixels[0]], [pixels[1]])
    pixels = (pixels[0] - point[2] * v0, pixels[1] - point[2] * v1)

    return pixels


def get_point_flow_vector(video, point, flow_parameters, crop_size=(21, 21)):
    startx0 = int(point[0]) - int(crop_size[0] / 2)
    endx0 = startx0 + crop_size[0]
    starty0 = int(point[1]) - int(crop_size[1] / 2)
    endy0 = starty0 + crop_size[1]

    start_frame = video[0][startx0:endx0, starty0:endy0]
    x_flows = [0]
    y_flows = [0]
    for frame_num, frame in enumerate(video[1:]):
        exp_diffx = int(x_flows[frame_num])
        exp_diffy = int(y_flows[frame_num])

        startx = startx0 + exp_diffx
        endx = endx0 + exp_diffx
        starty = starty0 + exp_diffy
        endy = endy0 + exp_diffy

        cur_frame = frame[startx:endx, starty:endy]

        flow_dictionary = flow_parameters.copy()
        # add remaining parameters
        flow_dictionary["prev"] = start_frame
        flow_dictionary["next"] = cur_frame
        flow_dictionary["flow"] = None
        flow = cv2.calcOpticalFlowFarneback(**flow_dictionary)

        center = (int(crop_size[0] / 2), int(crop_size[1] / 2))

        x_flow = flow[center[0], center[1], 1]
        y_flow = flow[center[0], center[1], 0]

        x_flow = x_flow + exp_diffx
        y_flow = y_flow + exp_diffy
        x_flows.append(x_flow)
        y_flows.append(y_flow)

    return x_flows, y_flows


# alternate method to get the flow vectors
# where i just look at small sections of the videos,
# while cropping around where the point moved in last frame
# basically this should make better tracking for videos where the ant moves a lot
def get_flow_vectors(videos, match_points, flow_parameters, crop_size=(21, 21)):
    num_cameras = len(match_points)
    num_points = len(match_points[0])
    num_frames = len(videos[0])
    flow_vectors = np.zeros((num_points, num_cameras * 2, num_frames))

    for cam_num, video in videos.items():
        points = match_points[cam_num]

        x_flows = np.zeros((num_points, num_frames))
        y_flows = x_flows.copy()
        for point_num, point in enumerate(points):
            point_x_flows, point_y_flows = get_point_flow_vector(
                video, point, flow_parameters, crop_size=crop_size
            )
            x_flows[point_num] = point_x_flows
            y_flows[point_num] = point_y_flows

        flow_vectors[:, 2 * cam_num] = x_flows
        flow_vectors[:, 2 * cam_num + 1] = y_flows
    return flow_vectors


def gaussian_kernel(size: int, sigma: float):
    """Creates a 1D Gaussian kernel."""
    x = torch.arange(-size // 2 + 1.0, size // 2 + 1.0)
    kernel = torch.exp(-0.5 * (x / sigma) ** 2)
    kernel = kernel / kernel.sum()
    return kernel


# original_points and transformed_points should have same shape
# B X num_points X 3
# note that Pytorch and numpy do seem to give slightly different results for SVD
# this is based on a function written by ChatGPT
def estimate_affine_transform(original_points, transformed_points):
    centroid_original = torch.mean(original_points, axis=0)
    centroid_transformed = torch.mean(transformed_points, axis=0)

    zero_centered_original = original_points - centroid_original
    zero_centered_transformed = transformed_points - centroid_transformed

    A = zero_centered_original.permute(1, 2, 0)
    B = zero_centered_transformed.permute(1, 0, 2)
    H = torch.bmm(A, B)

    U, _, V = torch.svd(H)

    rotation_matrix = torch.bmm(V, U.permute(0, 2, 1))
    translation = (
        centroid_transformed
        - torch.bmm(rotation_matrix, centroid_original.unsqueeze(-1)).squeeze()
    )
    affine_matrices = torch.zeros((translation.shape[0], 4, 4))
    affine_matrices[:, -1, -1] = 1
    affine_matrices[:, :3, :3] = rotation_matrix
    affine_matrices[:, :3, 3] = translation

    return affine_matrices


def get_global_movement(affine_matrices):
    global_movement = {}
    rotation_matrices = affine_matrices[:, :3, :3]
    global_movement["yaw"] = torch.arctan2(
        rotation_matrices[:, 1, 0], rotation_matrices[:, 0, 0]
    )
    global_movement["pitch"] = torch.arctan2(
        -rotation_matrices[:, 2, 0],
        torch.sqrt(rotation_matrices[:, 2, 1] ** 2 + rotation_matrices[:, 2, 2] ** 2),
    )
    global_movement["roll"] = torch.arctan2(
        rotation_matrices[:, 2, 1], rotation_matrices[:, 2, 2]
    )
    global_movement["x"] = affine_matrices[:, 0, 3]
    global_movement["y"] = affine_matrices[:, 1, 3]
    global_movement["z"] = affine_matrices[:, 2, 3]
    return global_movement


# define modules
class gaussian_filter(nn.Module):
    def __init__(self, size, sigma):
        super().__init__()
        filters = gaussian_kernel(size=size, sigma=sigma).view(1, 1, -1)
        self.register_parameter(name="filters", param=torch.nn.Parameter(filters))

    def forward(self, signal):
        assert len(signal.shape) == 3
        output = conv1d(signal, self.filters, padding="same")
        return output


class ModifiedHuber(nn.Module):
    def __init__(self, base_delta):
        super().__init__()
        self.base_delta = base_delta

    def forward(self, x, y, weights=None):
        if weights is None:
            weights = torch.ones_like(x)
        base_delta = self.base_delta
        r = torch.abs(x - y)
        loss = torch.ones_like(r)
        low_indices = r < weights * base_delta
        high_indices = ~low_indices

        loss[high_indices] = (
            weights[high_indices]
            * base_delta
            * (r[high_indices] - 0.5 * weights[high_indices] * base_delta)
        )
        loss[low_indices] = 0.5 * (r[low_indices] ** 2)
        return loss


# coefficients should be #points x 6 x 3
class ParallelLinear(nn.Module):
    def __init__(self, coefficients, num_frames):
        super().__init__()
        num_points = coefficients.shape[0]
        self.register_buffer("coefficients", coefficients)
        self.register_parameter(
            name="displacements",
            param=torch.nn.Parameter(torch.randn((num_points, 3, num_frames))),
        )

    def forward(self):
        predicted_flows = torch.bmm(self.coefficients, self.displacements)
        return predicted_flows

    def cuda(self):
        self.coefficients.cuda()
        return super().cuda()


# older functions

camera_combos = {"0, 1": [0, 1], "0, 2": [0, 2], "1, 2": [1, 2]}


# this function will likely be re-done, and is currently hard-coded for a three-camera system
# just using pairs of cameras to get the locations
def get_point_locations(
    system, match_points, remove_imprecise_vals=True, return_average=True
):
    first_key = [i for i in match_points.keys()][0]
    num_points = len(match_points[first_key])
    location_estimates_mm = xr.DataArray(
        np.ones((num_points, 3, 4), dtype=np.float64) * np.nan,
        dims=("point_number", "camera_combo", "spatial_dim"),
        coords={
            "point_number": np.arange(len(match_points[0])),
            "camera_combo": ["0, 1", "0, 2", "1, 2"],
            "spatial_dim": ["x", "y", "z_x", "z_y"],
        },
    )

    for key, cameras in camera_combos.items():
        cam_num0 = cameras[0]
        cam_num1 = cameras[1]

        points_cam0 = match_points[cam_num0]
        points_cam1 = match_points[cam_num1]
        for point_number, (point_cam0, point_cam1) in enumerate(
            zip(points_cam0, points_cam1)
        ):
            # plane0 is computed using the difference in the 0 dimension
            # plane1 is computed using the difference in the 1 dimension
            # both of these values will be in mm

            plane_from_dx, plane_from_dy = system.plane_from_pixel_locs(
                cam_number0=cam_num0,
                cam_number1=cam_num1,
                pixel_loc0=(point_cam0[0], point_cam0[1]),
                pixel_loc1=(point_cam1[0], point_cam1[1]),
            )
            # then we need to find x and y from that.
            # for now, we're finding the true x and y locations in "pixels"
            # so where that point would appear in the image if it was translated to the reference plane, plane=0
            # we can use either camera for this
            slope0_cam0, slope1_cam0 = system.get_shift_slopes(
                cam_num0, [point_cam0[0]], [point_cam0[1]]
            )
            # the 0 indexing on slope0_cam0 and others is because get_shift_slopes returns a list
            x = point_cam0[0] - slope0_cam0[0] * plane_from_dx
            y = point_cam0[1] - slope1_cam0[0] * plane_from_dy

            # Then shift the pixel values to their location in the reference camera
            # ideally we would use x0, x1, y0, y1 instead of point_cam0 and point_cam1
            # because inter-camera shifts are measured at the reference plane
            # but we do it this way to prevent large height estimation errors in plane_from_dx OR plane_from_dy
            # from affecting estimations in the other dimension
            shiftx_cam0, shifty_cam0 = system.get_pixel_shifts(
                cam_num0, [point_cam0[0]], [point_cam0[1]]
            )
            x = x + shiftx_cam0[0]
            y = y + shifty_cam0[0]

            # to switch to mm, use the magnificaiton at the reference camera
            # since we would've already taken into account differences in magnificaiton
            # in the shift to the reference camera
            pixel_size_m = system.calib_manager.pixel_size
            magnification_dim0 = system.get_magnification_at_plane(
                camera_number=system.reference_camera, plane_mm=0, dim=0
            )
            magnification_dim1 = system.get_magnification_at_plane(
                camera_number=system.reference_camera, plane_mm=0, dim=1
            )
            x_mm = x / magnification_dim0 * pixel_size_m * 1e3
            y_mm = y / magnification_dim1 * pixel_size_m * 1e3

            entry_dict = {
                "x": x_mm,
                "y": y_mm,
                "z_x": plane_from_dx,
                "z_y": plane_from_dy,
            }
            for dim, entry in entry_dict.items():
                location_estimates_mm.loc[
                    {
                        "point_number": point_number,
                        "camera_combo": key,
                        "spatial_dim": dim,
                    }
                ] = entry

    if remove_imprecise_vals:
        location_estimates_mm.loc[
            {"spatial_dim": ["z_x", "x"], "camera_combo": "0, 1"}
        ] = np.nan

    if not return_average:
        return location_estimates_mm

    # then average to get the final values
    locations_mm = location_estimates_mm.mean(dim=["camera_combo"])
    # z has to be averaged further
    # but this could all be much cleaner
    z_locations_mm = (
        np.mean(locations_mm.sel(spatial_dim=["z_x", "z_y"]).data, axis=1) * -1
    )
    x_locations_mm = locations_mm.sel(spatial_dim="x").data
    y_locations_mm = locations_mm.sel(spatial_dim="y").data

    camera_points = np.concatenate(
        (x_locations_mm[:, None], y_locations_mm[:, None], z_locations_mm[:, None]),
        axis=1,
    )

    return camera_points


# default_flow_parameters = {
#     "pyr_scale": 0.5,
#     "levels": 3,
#     "winsize": 11,
#     "iterations": 5,
#     "poly_n": 5,
#     "poly_sigma": 0.8,
#     "flags": 0,
# }

default_flow_parameters = {
    "pyr_scale": 0.9,
    "levels": 5,
    "winsize": 21,
    "iterations": 5,
    "poly_n": 3,
    "poly_sigma": 0.8,
    "flags": 0,
}


# this is primarily used to get new start point locations
# for a video, based off a previous alignment image
def match_points_between_images(prev_image, new_image, points, flow_parameters=None):
    if flow_parameters is None:
        flow_parameters = default_flow_parameters

    flow_dict = flow_parameters

    buffer = 11
    minx = int(np.min(points[:, 0]) - buffer)
    maxx = int(np.max(points[:, 0]) + buffer)
    miny = int(np.min(points[:, 1]) - buffer)
    maxy = int(np.max(points[:, 1]) + buffer)

    prev_image = prev_image[minx:maxx, miny:maxy]
    new_image = new_image[minx:maxx, miny:maxy]

    flow_dict["prev"] = prev_image
    flow_dict["next"] = new_image
    flow_dict["flow"] = None

    flow = cv2.calcOpticalFlowFarneback(**flow_dict)
    mp = points - [minx, miny]
    mp = mp.astype(int)
    flow_values = flow[mp[:, 0], mp[:, 1]]

    new_points = points + flow_values
    return new_points


if __name__ == "__main__":
    from hsflfm.util import MetadataManager, load_dictionary, procrustes_analysis
    from hsflfm.calibration import FLF_System
    from matplotlib import pyplot as plt
    from hsflfm.processing import Aligner

    specimen_number = "20220427_OB_4"
    manager = MetadataManager(specimen_number)

    strike_match_points = {}
    for cam in range(3):
        prev_image = manager.get_start_images(strike_number=1)[cam]
        new_image = manager.get_start_images(strike_number=3)[cam]

        match_points = np.asarray(load_dictionary(manager.match_points_filename)[cam])[
            :, :2
        ]
        new_points = match_points_between_images(prev_image, new_image, match_points)

        strike_match_points[cam] = new_points

    strike_locations = get_point_locations(
        FLF_System(manager.calibration_filename), strike_match_points
    )
    match_points = load_dictionary(manager.match_points_filename)
    locations = get_point_locations(
        FLF_System(manager.calibration_filename), match_points
    )

    A_cam2_to_cam1, _, transformed_points = procrustes_analysis(
        strike_locations,
        locations,
        allow_scale=False,
    )

    diff = locations - transformed_points
    distances = np.linalg.norm(diff, axis=1)

    aligner = Aligner(specimen_number)
    A, smp, pn, bn = aligner.align_strike(5)
