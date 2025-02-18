from hsflfm.ant_model import (
    mesh_filename,
    mesh_with_mandibles_filename,
    mesh_scale,
    display_mesh_scale,
)
from hsflfm.config import home_directory
from .bulk_analyzer import ResultManager
from hsflfm.processing import world_frame_to_pixel
from hsflfm.util import load_split_video, MetadataManager
from hsflfm.calibration import FLF_System

import trimesh
import torch
from matplotlib import pyplot as plt
import matplotlib
import plotly.graph_objects as go
import numpy as np
from scipy.spatial import KDTree
from tqdm import tqdm

mesh_filename = home_directory + "/" + mesh_filename
display_mesh_filename = home_directory + "/" + mesh_with_mandibles_filename


class ResultPlotter:
    def __init__(self, result_info, good_indices=None):
        self.result_info = result_info
        self.data_manager = MetadataManager(self.result_info["specimen_number"])
        self.result_manager = ResultManager(self.result_info)
        self.videos = None
        self.strike_number = self.result_info["strike_number"]

        self.peak_indices = self.result_manager.peak_indices(dim=2)
        self.strike_center = self.result_manager.strike_center_index()

        self.system = FLF_System(self.data_manager.calibration_filename)

        # this can be set at any point to change
        # which set of points get used in displays
        self.good_indices = good_indices

    def load_video(self):
        self.videos = load_split_video(
            self.data_manager.video_filename(self.result_info["strike_number"]),
            calibration_filename=self.data_manager.calibration_filename,
        )
        # ensure grayscale
        for key, item in self.videos.items():
            if len(item.shape) > 3:
                item = np.mean(item, axis=-1)
            self.videos[key] = item

    def scatter_over_image(
        self,
        values,
        highlight_point=None,
        cam=2,
        good_only=True,
        crop=True,
        *args,
        **kwargs,
    ):
        img = self.data_manager.get_start_images(strike_number=self.strike_number)[cam]
        mp = torch.asarray(self.result_info["match_points"][cam])

        if self.good_indices is not None and good_only:
            mp = mp[self.good_indices]

        fig = plt.figure()
        plt.imshow(img, cmap="gray")
        plt.scatter(mp[:, 1], mp[:, 0], c=values, *args, **kwargs)

        if crop:
            buffer = 22
            startx = torch.min(mp[:, 0]) - buffer
            endx = torch.max(mp[:, 0]) + buffer
            starty = torch.min(mp[:, 1]) - buffer
            endy = torch.max(mp[:, 1]) + buffer
            plt.ylim(endx, startx)
            plt.xlim(starty, endy)

        if highlight_point is not None:
            point_numbers = torch.asarray(self.result_info["point_numbers"])
            idx = torch.where(point_numbers == highlight_point)[0][0]
            loc = mp[idx]
            circ = matplotlib.patches.Circle(
                (loc[1], loc[0]), 3, color="white", fill=False
            )
            plt.gca().add_patch(circ)
        return fig

    def scatter_values(
        self, values, highlight_point=None, good_only=True, *args, **kwargs
    ):
        ant_start_locs = self.result_manager.point_start_locs_ant_mm

        if self.good_indices is not None and good_only:
            ant_start_locs = ant_start_locs[self.good_indices]

        fig = plt.figure()
        plt.scatter(
            ant_start_locs[:, 1], ant_start_locs[:, 0], c=values, *args, **kwargs
        )

        if highlight_point is not None:
            point_numbers = torch.asarray(self.result_info["point_numbers"])
            idx = torch.where(point_numbers == highlight_point)[0][0]
            loc = ant_start_locs[idx][:2]
            circ = matplotlib.patches.Circle(
                (loc[1], loc[0]), 0.03, color="black", fill=False
            )
            plt.gca().add_patch(circ)

        plt.ylabel("x")
        plt.xlabel("y")
        plt.gca().invert_xaxis()
        plt.gca().set_aspect("equal")
        plt.colorbar()
        return fig

    def show_image_numbers(self, cam=2, good_only=True):
        point_numbers = torch.asarray(self.result_info["point_numbers"])
        fig = self.scatter_over_image(values="black", cam=cam)

        ax = fig.axes[0]
        mp = np.asarray(self.result_info["match_points"][cam])

        if good_only and self.good_indices is not None:
            point_numbers = point_numbers[self.good_indices]
            mp = mp[self.good_indices]

        for number, point in zip(point_numbers, mp):
            ax.text(point[1], point[0], str(int(number)), color="white")

        return fig

    def scatter_peak_disp(
        self, dim=2, with_image=False, good_only=True, *args, **kwargs
    ):
        peak_displacement = self.result_manager.peak_displacements(dim=dim)

        if good_only and self.good_indices is not None:
            peak_displacement = peak_displacement[self.good_indices]

        if with_image:
            self.scatter_over_image(
                peak_displacement, good_only=good_only, *args, **kwargs
            )
        else:
            self.scatter_values(peak_displacement, good_only=good_only, *args, **kwargs)

    def plot_camera_weight(self, point_number):
        point_index = torch.where(
            torch.asarray(self.result_info["point_numbers"]) == point_number
        )[0][0]
        weights = torch.asarray(self.result_info["camera_weights"])[point_index]

        fig = plt.figure()
        for i, (p, color) in enumerate(zip(weights, ["red", "blue", "green"])):
            plt.plot(p, label=f"cam {i}", color=color)

        plt.xlabel("Frame #")
        plt.ylabel("Camera Weight")
        plt.axvline(
            x=self.peak_indices[point_index],
            linestyle="--",
            color="black",
            label="Peak Point",
        )
        plt.legend()
        return fig

    def plot_displacement(self, point_number):
        point_index = torch.where(
            torch.asarray(self.result_info["point_numbers"]) == point_number
        )[0][0]
        rel_displacements = self.result_manager.rel_displacements[point_index]
        abs_displacements = torch.asarray(
            self.result_info["camera_point_displacements"]
        )[point_index]

        fig, (ax0, ax1) = plt.subplots(1, 2)
        for i, dim in enumerate(["x", "y", "z"]):
            ax0.plot(rel_displacements[:, i] * 1e3, label=dim)
            ax1.plot(abs_displacements[:, i] * 1e3)
        ax0.legend()

        for ax in [ax0, ax1]:
            ax.set_xlabel("Frame #")
            ax.set_ylabel("Displacement (um)")
        plt.tight_layout()
        return fig

    def show_flow_differences(self, point_number, show_peak_index=True):
        point_index = torch.where(
            torch.asarray(self.result_info["point_numbers"]) == point_number
        )[0][0]

        peak_index = self.peak_indices[point_index]

        colors2 = [
            ["#FF00FF", "#C20078"],
            ["#0000FF", "#0343DF"],
            ["#008000", "#15B01A"],
        ]

        predictions = torch.asarray(self.result_info["predicted_flow_vectors"])
        flow_vectors = torch.asarray(self.result_info["flow_vectors"])
        diff = self.result_manager.flow_differences

        fig0, ax0 = plt.subplots(1, 1)
        fig1, ax1 = plt.subplots(1, 1)

        for i, (p0, p1) in enumerate(
            zip(flow_vectors[point_index], predictions[point_index])
        ):
            cam_num = int(i / 2)
            dim = i % 2
            color = colors2[cam_num][dim]
            ax0.plot(p0, alpha=0.5, color=color)
            label = f"cam {cam_num}, dim {dim}"
            ax0.plot(p1, color=color, label=label)

            ax1.plot(torch.abs(diff[point_index, i]), color=color, label=label)

        ax1.plot(
            torch.mean(torch.abs(diff[point_index]), axis=0),
            color="black",
            label="average",
        )

        if show_peak_index:
            ax0.axvline(x=peak_index, color="black", linestyle="--", label="Peak Point")
            ax1.axvline(x=peak_index, color="black", linestyle="--", label="Peak Point")
        ax0.legend(bbox_to_anchor=(1, 0.5))
        # ax1.legend(bbox_to_anchor=(1, 0.5))

        ax1.set_ylabel("abs(Flow Residual) (pixels)")
        ax0.set_ylabel("Optical Flow (pixels)")
        ax1.set_xlabel("Frame #")
        ax0.set_xlabel("Frame #")

        return fig0, fig1

    def plot_all_displacement(
        self, dim=2, relative=True, highlight_point=None, good_only=True
    ):
        if relative:
            disp = torch.asarray(self.result_info["rel_displacements"])
        else:
            disp = torch.asarray(self.result_info["camera_point_displacements"])
        disp = disp[:, :, dim] * 1e3

        if good_only and self.good_indices is not None:
            disp_ = disp[self.good_indices]
        else:
            disp_ = disp

        fig = plt.figure()
        for p in disp_:
            plt.plot(p, color="black")
        if highlight_point is not None:
            point_index = torch.where(
                torch.asarray(self.result_info["point_numbers"]) == highlight_point
            )[0][0]
            plt.plot(disp[point_index], color="red", label=f"point {highlight_point}")
            plt.legend()
        plt.xlabel("Frame #")
        plt.ylabel("Displacement (um)")
        return fig

    def make_point_track_video(
        self,
        cam_num=2,
        crop_buffer=[75, 10, 25, 25],
        highlight_point=None,
        frames=None,
        return_crops=False,
        white_buffer=5,
    ):
        if highlight_point is not None:
            point_index = torch.where(
                torch.asarray(self.result_info["point_numbers"]) == highlight_point
            )[0][0]
        if self.videos is None:
            self.load_video()

        # can specify to only make the video for certain frames
        if frames is None:
            frames = torch.arange(self.videos[cam_num].shape[0])

        video = None
        xs = None
        ys = None
        xe = None
        ye = None
        fig = plt.figure()

        flow = torch.asarray(self.result_info["flow_vectors"])
        flowx = flow[:, 2 * cam_num]
        flowy = flow[:, 2 * cam_num + 1]
        match_points = torch.asarray(self.result_info["match_points"][cam_num])
        if crop_buffer is not None:
            xstart = torch.min(match_points[:, 0]) - crop_buffer[0]
            xend = torch.max(match_points[:, 0]) + crop_buffer[1]
            ystart = torch.min(match_points[:, 1]) - crop_buffer[2]
            yend = torch.max(match_points[:, 1]) + crop_buffer[3]

        gray_video = self.videos[cam_num]
        # if scale_gray_vid:
        #     gray_video = (gray_video - np.min(gray_video)) / (
        #         np.max(gray_video) - np.min(gray_video)
        #     )
        #     gray_video = (gray_video * 255).astype(np.uint8)
        # gray_video = (gray_video * video_brightness).astype(np.uint8)

        for i, frame_num in tqdm(enumerate(frames)):
            # hack hack hack
            # showing fake image to get the colorbar ?
            frame = torch.asarray(gray_video[frame_num]).to(torch.uint8)
            plt.imshow(frame, cmap="gray")
            plt.xticks([])
            plt.yticks([])

            if crop_buffer is not None:
                plt.gca().set_xlim(ystart, yend)
                plt.gca().set_ylim(xend, xstart)

            points = np.asarray(match_points).copy()
            points[:, 0] = points[:, 0] + flowx[:, frame_num].numpy()
            points[:, 1] = points[:, 1] + flowy[:, frame_num].numpy()
            plt.scatter(points[:, 1], points[:, 0], color="blue")

            if highlight_point is not None:
                plt.scatter(
                    [points[point_index, 1]], [points[point_index, 0]], color="red"
                )

            plt.tight_layout()
            fig.canvas.draw()
            plot_array = np.asarray(fig.canvas.renderer._renderer)

            if video is None:
                a = np.mean(plot_array, axis=-1)
                x, y = np.where(a != 255)
                xs = max(0, np.min(x) - white_buffer)
                xe = min(plot_array.shape[0], np.max(x) + white_buffer)
                ys = max(0, np.min(y) - white_buffer)
                ye = min(plot_array.shape[1], np.max(y) + white_buffer)
                plot_array = plot_array[xs:xe, ys:ye]
                video = np.zeros(
                    (
                        len(frames),
                        plot_array.shape[0],
                        plot_array.shape[1],
                        plot_array.shape[2],
                    ),
                    dtype=np.uint8,
                )
            else:
                plot_array = plot_array[xs:xe, ys:ye]

            video[i] = plot_array
            fig.clear()
        if return_crops:
            crops = [int(i) for i in [xstart, xend, ystart, yend]]
            return video, crops
        return video

    # make video with arrows annotating point movement
    # this can likely be simplified using newer functions for getting point locations
    def get_arrow_video(
        self,
        disp_threshold=2e-3,
        cam_num=2,
        show_coord_frame=True,
        movement_mag=15,
        crop_buffer=[75, 10, 25, 25],
        cmap=matplotlib.cm.turbo,
        frames=None,
        force_arrow_after_strike=True,
        return_crops=False,
        white_buffer=5,
        good_only=True,
    ):
        if self.videos is None:
            self.load_video()

        # can specify to only make the video for certain frames
        if frames is None:
            frames = torch.arange(self.videos[cam_num].shape[0])

        rel_displacements = torch.asarray(self.result_info["rel_displacements"])
        affine_matrices = torch.asarray(self.result_info["affine_matrices"])
        A_cam_to_ant_start = torch.asarray(self.result_info["A_cam_to_ant_start"])
        match_points = torch.asarray(self.result_info["match_points"][cam_num])

        if good_only and self.good_indices is not None:
            idx = self.good_indices
        else:
            idx = torch.arange(rel_displacements.shape[0])

        max_z_disp = torch.max(rel_displacements[idx, :, 2])
        min_z_disp = torch.min(rel_displacements[idx, :, 2])

        flow = torch.asarray(self.result_info["flow_vectors"])
        flowx = flow[:, 2 * cam_num]
        flowy = flow[:, 2 * cam_num + 1]

        video = None
        xs = None
        ys = None
        xe = None
        ye = None
        fig = plt.figure()

        if crop_buffer is not None:
            xstart = torch.min(match_points[:, 0]) - crop_buffer[0]
            xend = torch.max(match_points[:, 0]) + crop_buffer[1]
            ystart = torch.min(match_points[:, 1]) - crop_buffer[2]
            yend = torch.max(match_points[:, 1]) + crop_buffer[3]

        gray_video = self.videos[cam_num]
        # if scale_gray_vid:
        #     gray_video = (gray_video - np.min(gray_video)) / (
        #         np.max(gray_video) - np.min(gray_video)
        #     )
        #     gray_video = (gray_video * 255).astype(np.uint8)
        # gray_video = (gray_video * video_brightness).astype(np.uint8)
        for i, frame_num in tqdm(enumerate(frames)):
            # hack hack hack
            # showing fake image to get the colorbar ?
            frame = torch.asarray(gray_video[frame_num]).to(torch.uint8)
            plt.imshow(
                (
                    torch.rand((frame.shape[0], frame.shape[1]))
                    * (max_z_disp - min_z_disp)
                    + min_z_disp
                )
                * 1e3,
                cmap=cmap,
            )
            plt.colorbar()
            plt.imshow(frame, cmap="gray")
            plt.xticks([])
            plt.yticks([])

            if crop_buffer is not None:
                plt.gca().set_xlim(ystart, yend)
                plt.gca().set_ylim(xend, xstart)

            A_frame_to_start = affine_matrices[frame_num]
            A_ant_to_cam = torch.matmul(
                torch.linalg.inv(A_cam_to_ant_start), A_frame_to_start
            )

            start = world_frame_to_pixel(
                self.system,
                torch.matmul(
                    A_ant_to_cam, torch.asarray([0, 0, 0, 1], dtype=torch.float32)
                ),
            )
            scale = 0.35
            v0 = world_frame_to_pixel(
                self.system,
                torch.matmul(
                    A_ant_to_cam,
                    torch.asarray((1 * scale, 0, 0, 1), dtype=torch.float32),
                ),
            )
            v1 = world_frame_to_pixel(
                self.system,
                torch.matmul(
                    A_ant_to_cam,
                    torch.asarray((0, 1 * scale, 0, 1), dtype=torch.float32),
                ),
            )

            if show_coord_frame:
                plt.arrow(
                    start[1][0],
                    start[0][0],
                    v0[1][0] - start[1][0],
                    v0[0][0] - start[0][0],
                    color="blue",
                    head_width=2,
                )
                plt.arrow(
                    start[1][0],
                    start[0][0],
                    v1[1][0] - start[1][0],
                    v1[0][0] - start[0][0],
                    color="blue",
                    head_width=2,
                )

            x_vect = (v0[0][0] - start[0][0], v0[1][0] - start[1][0])
            y_vect = (v1[0][0] - start[0][0], v1[1][0] - start[1][0])

            for point_index in range(match_points.shape[0]):
                if good_only and self.good_indices is not None:
                    if point_index not in self.good_indices:
                        continue
                disp = rel_displacements[point_index, frame_num]
                z_disp = disp[2]

                norm_z_disp = (z_disp - min_z_disp) / (max_z_disp - min_z_disp)
                max_z_disp = max(max_z_disp, z_disp)
                min_z_disp = min(min_z_disp, z_disp)

                point = match_points[point_index]
                x_flow = flowx[point_index][frame_num]
                y_flow = flowy[point_index][frame_num]

                color = cmap(norm_z_disp)

                point = [point[0] + x_flow, point[1] + y_flow]

                arrow_scale = movement_mag / scale
                x_mag = (x_vect[0] * disp[0] + y_vect[0] * disp[1]) * arrow_scale
                y_mag = (x_vect[1] * disp[0] + y_vect[1] * disp[1]) * arrow_scale

                use_arrow = (
                    (disp_threshold is None)
                    or (torch.sqrt(disp[0] ** 2 + disp[1] ** 2) > disp_threshold)
                    or (force_arrow_after_strike and frame_num > self.strike_center)
                )
                if use_arrow:
                    plt.arrow(
                        point[1] - y_mag / 2,
                        point[0] - x_mag / 2,
                        y_mag,
                        x_mag,
                        color=color,
                        head_width=2,
                    )
                else:
                    plt.scatter([point[1]], [point[0]], color=color, s=20)
            plt.tight_layout()
            fig.canvas.draw()
            plot_array = np.asarray(fig.canvas.renderer._renderer)

            if video is None:
                a = np.mean(plot_array, axis=-1)
                x, y = np.where(a != 255)
                xs = max(0, np.min(x) - white_buffer)
                xe = min(plot_array.shape[0], np.max(x) + white_buffer)
                ys = max(0, np.min(y) - white_buffer)
                ye = min(plot_array.shape[1], np.max(y) + white_buffer)
                plot_array = plot_array[xs:xe, ys:ye]
                video = np.zeros(
                    (
                        len(frames),
                        plot_array.shape[0],
                        plot_array.shape[1],
                        plot_array.shape[2],
                    ),
                    dtype=np.uint8,
                )
            else:
                plot_array = plot_array[xs:xe, ys:ye]

            video[i] = plot_array
            fig.clear()
        if return_crops:
            crops = [int(i) for i in [xstart, xend, ystart, yend]]
            return video, crops
        return video

    # make video with plane showing global movement

    # make video that plots out trajectories of certain points

    @staticmethod
    def move_points_to_surface(mesh_points, mesh=None, mesh_sample=100):
        if mesh is None:
            mesh = trimesh.load(mesh_filename)
        vertices, _ = trimesh.sample.sample_surface(
            mesh, len(mesh.vertices) * mesh_sample
        )
        tree = KDTree(vertices)
        _, indices = tree.query(mesh_points)
        return vertices[indices]

    @staticmethod
    def plot_mesh_with_points(
        mesh=None,
        opacity=0.5,
        color="lightgray",
        points=None,
        point_values=None,
        points_on_surface=False,
        use_display_mesh=True,
        marker_dict={"size": 5, "opacity": 1, "colorscale": "Turbo"},
        show=True,
    ):
        if points_on_surface:
            points = ResultPlotter.move_points_to_surface(points)

        if mesh is None:
            if use_display_mesh:
                mesh = trimesh.load(display_mesh_filename)
                scale_ratio = display_mesh_scale / mesh_scale
            else:
                mesh = trimesh.load(mesh_filename)
                scale_ratio = 1
            # we're using the display mesh, so the point values need to change
            if points is not None:
                points = points * scale_ratio

        mesh_x = mesh.vertices[:, 0]
        mesh_y = mesh.vertices[:, 1]
        mesh_z = mesh.vertices[:, 2]
        mesh_faces = mesh.faces
        fig = go.Figure(
            data=[
                go.Mesh3d(
                    x=mesh_x,
                    y=mesh_y,
                    z=mesh_z,
                    i=mesh_faces[:, 0],
                    j=mesh_faces[:, 1],
                    k=mesh_faces[:, 2],
                    opacity=opacity,
                    color=color,
                )
            ]
        )

        if points is not None:
            if point_values is None:
                point_values = "red"
            marker_dict["color"] = point_values
            fig.add_trace(
                go.Scatter3d(
                    x=points[:, 0],
                    y=points[:, 1],
                    z=points[:, 2],
                    mode="markers",
                    marker=marker_dict,
                )
            )

        if show:
            fig.show()

        return fig
