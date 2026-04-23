# this is a utility class to easily manage metadata information about a given ant
# this will prevent having to frequently interact directly with the metadata file
from hsflfm.config import home_directory, metadata_filename, data_folder
from hsflfm.util import load_split_video, load_image_set

import numpy as np
import pandas as pd

metadata = pd.read_csv(home_directory + "/" + metadata_filename)


class MetadataManager:
    # the code treats each alignment setup as a different specimen
    # in the main dataset, only one specimen, "20240418_OB_1" has two alignment setups
    def __init__(self, specimen_number):
        self.alignment_tag = specimen_number
        self.specimen_data = metadata.loc[metadata["Alignment Tag"] == specimen_number]
        if len(self.specimen_data) == 0:
            raise ValueError(f"Specimen number {specimen_number} not found in metadata")

    @staticmethod
    def all_specimen_numbers():
        return np.unique(metadata["Alignment Tag"].values)

    @property
    def strike_numbers(self):
        return np.sort(self.specimen_data["Strike #"].values)

    @property
    def main_folder(self):
        main_folder = self.specimen_data["Main Folder"].values[0]
        return data_folder + "/" + str(main_folder)

    @property
    def video_folder(self):
        folder = self.main_folder + f"/{self.alignment_tag}"
        return folder

    # where the calibration images are stored
    @property
    def calibration_folder(self):
        return self.main_folder + "/calibration_images"

    @property
    def calibration_filename(self):
        filename = self.main_folder + "/calibration_information.json"
        return filename

    @property
    def alignment_folder(self):
        return self.video_folder + "/alignment_files"

    def get_strike_data(self, strike_number):
        return self.specimen_data.loc[self.specimen_data["Strike #"] == strike_number]

    def get_frame_rate(self, strike_number):
        return self.get_strike_data(strike_number)["Frame Rate"].values[0]

    def video_filename(self, strike_number):
        video_folder = self.video_folder
        strike_data = self.get_strike_data(strike_number)
        video_filename = strike_data["Video Filename"].values[0]
        return video_folder + "/" + video_filename

    @property
    def match_points_filename(self):
        folder = self.alignment_folder
        return folder + "/match_points.json"

    @property
    def alignment_points_filename(self):
        folder = self.alignment_folder
        return folder + "/alignment_points.json"

    @property
    def oblique_alignment_filename(self):
        folder = self.alignment_folder
        filename = f"{self.alignment_tag}_oblique_alignment.tiff"
        return folder + "/" + filename

    @property
    def alignment_image_filename(self):
        folder = self.alignment_folder
        filename = f"{self.alignment_tag}_alignment.tiff"
        return folder + "/" + filename

    @property
    def oblique_alignment_images(self):
        filename = self.oblique_alignment_filename
        images = load_image_set(filename, self.calibration_filename)
        return images

    @property
    def alignment_images(self):
        filename = self.alignment_image_filename
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
            # don't use the first frame because they're sometimes weird
            frame = np.mean(video[1:], axis=0)
            images[key] = frame
        return images

    # these could likely be handled in a better way
    # but this is okay for now
    def mandible_order(self, strike_number):
        strike_data = self.get_strike_data(strike_number)
        key = "Mandible Order"
        return strike_data[key].values[0].strip()
