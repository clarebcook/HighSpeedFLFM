# this is a utility class to easily manage metadata information about a given ant
# this will prevent having to frequently interact directly with the metadata file
from hsflfm.config import home_directory, metadata_filename
from hsflfm.util import load_split_video, load_image_set

import numpy as np
import pandas as pd

metadata = pd.read_excel(home_directory + "/" + metadata_filename)


class MetadataManager:
    def __init__(self, specimen_number):
        self.specimen_data = metadata.loc[metadata["Specimen #"] == specimen_number]

    @staticmethod
    def all_specimen_numbers():
        return np.unique(metadata["Specimen #"].values)

    @property
    def strike_numbers(self):
        return np.sort(self.specimen_data["Strike #"].values)

    @property
    def calibration_folder(self):
        folder = self.specimen_data["Calibration Folder"].values[0]
        return home_directory + "/Videos/" + folder

    @property
    def video_folder(self):
        folder = self.specimen_data["Video Folder"].values[0]
        return home_directory + "/Videos/" + folder

    @property
    def calibration_filename(self):
        filename = self.calibration_folder + "/" + "calibration_information"
        return filename

    @property
    def alignment_folder(self):
        folder = self.specimen_data["Alignment Data Folder"].values[0]
        return home_directory + "/Videos/" + folder

    @property
    def alignment_image_folder(self):
        folder = self.specimen_data["Alignment Image Folder"].values[0]
        return home_directory + "/Videos/" + folder

    def get_strike_data(self, strike_number):
        return self.specimen_data.loc[self.specimen_data["Strike #"] == strike_number]

    def video_filename(self, strike_number):
        video_folder = self.video_folder
        strike_data = self.get_strike_data(strike_number)
        video_filename = strike_data["VideoFileName"].values[0]
        return video_folder + "/" + video_filename

    @property
    def match_points_filename(self):
        folder = self.alignment_folder
        return folder + "/match_points"

    @property
    def alignment_points_filename(self):
        folder = self.alignment_folder
        return folder + "/alignment_points"

    @property
    def light_calibration_filename(self):
        folder = self.alignment_image_folder
        filename = self.specimen_data["Alignment \n(side light)\n FileName"].values[0]
        return folder + "/" + filename

    @property
    def dark_calibration_filename(self):
        folder = self.alignment_image_folder
        filename = self.specimen_data["Alignment \n(ring light)\nFileName"].values[0]
        return folder + "/" + filename

    # this might not be the best place for these functions
    # but it's fine for now
    @property
    def light_calibration_images(self):
        filename = self.light_calibration_filename
        images = load_image_set(filename, self.calibration_filename)
        return images

    @property
    def dark_calibration_images(self):
        filename = self.dark_calibration_filename
        images = load_image_set(filename, self.calibration_filename)
        return images

    def get_start_images(self, strike_number, avg_frames=6):
        video_filename = self.video_filename(strike_number)
        gray_videos = load_split_video(
            video_filename, self.calibration_filename, max_frames=avg_frames + 1
        )
        images = {}
        for key, video in gray_videos.items():
            if len(video.shape) > 3:
                video = np.mean(video, axis=-1)
            # don't use the first frame because they're sometiems weird
            frame = np.mean(video[1:], axis=0)
            images[key] = frame
        return images

    # these could likely be handled in a better way
    # but this is okay for now
    def mandible_order(self, strike_number):
        strike_data = self.get_strike_data(strike_number)
        key = "Mandible Order \n(Left, Right, Simultaneous)"
        return strike_data[key].values[0].strip()

    # these will always be returned as L, R
    def mandible_start_frames(self, strike_number):
        strike_data = self.get_strike_data(strike_number)
        key0 = "Mandible 1 \nframe start / Strike Start"
        key1 = "Mandible 2 frame start"

        f0 = strike_data[key0].values[0]
        f1 = strike_data[key1].values[0]

        # TODO: edit this once we know how to handle "S"
        order = self.mandible_order(strike_number)
        if order in np.asarray(["L", "L only", "S"]):
            frames = (f0, f1)
        elif order in np.asarray(["R", "R only"]):
            frames = (f1, f0)
        else:
            raise ValueError(f"{order} not an expected order")

        return frames
