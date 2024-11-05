import numpy as np
import json
import tqdm as tqdm
import torch
import os
from PIL import Image 
import cv2

#### functions for loading and saving dictionaries


# makes sure integer keys are not of type np.int32 or np.int64
# this recusrively goes through possible multi-layer dictionary
def _clean_dict_keys_for_saving(dictionary):
    if not isinstance(dictionary, dict):
        return dictionary

    temp_dict = {}
    for key, item in dictionary.items():
        item = _clean_dict_keys_for_saving(item)
        if type(key) is np.int32 or type(key) is np.int64:
            temp_dict[int(key)] = item
        else:
            temp_dict[key] = item
    return temp_dict


# function to make sure every numpy array in a value or dict
# is converted to a list for saving
def _recursive_numpy_to_list(item):
    # if the item is a numpy array torch tensor , make it a list and return
    if isinstance(item, np.ndarray) or isinstance(item, torch.Tensor):
        return item.tolist()

    # if it's not a dictionary, return the item
    if not isinstance(item, dict):
        return item

    # if it is a dictionary, recursively call function for every entry
    for key, value in item.items():
        item[key] = _recursive_numpy_to_list(value)

    return item


# just some shortcut functions for loading/saving dictionaries
def save_dictionary(dictionary, save_filename, clean=True):
    if clean:
        dictionary = _clean_dict_keys_for_saving(dictionary)
        dictionary = _recursive_numpy_to_list(dictionary)
    with open(save_filename, "w") as fp:
        json.dump(dictionary, fp)


# dictionaries with integer keys get converted to strings
# this recursively changes those keys back to ints
def make_keys_ints(dictionary):
    if not isinstance(dictionary, dict):
        return dictionary

    temp_dict = {}
    for key, item in dictionary.items():
        item = make_keys_ints(item)
        if key.isdigit():
            temp_dict[int(key)] = item
        else:
            temp_dict[key] = item
    return temp_dict


def load_dictionary(filename, keys_are_ints=True):
    rawfile = open(filename)
    dictionary = json.load(rawfile)
    if keys_are_ints:
        dictionary = make_keys_ints(dictionary)
    return dictionary

def load_image_set(filename, calibration_filename=None, ensure_grayscale=True,
                   image_numbers=None):
    if calibration_filename is None:
        calibration_filename = (
            os.path.dirname(os.path.abspath(filename)) + "/calibration_information"
        )

    all_indices = load_dictionary(calibration_filename)["crop_indices"]
    # all_indices = CalibrationInfoManager(calibration_filename).crop_indices
    raw_image = Image.open(filename)
    image = np.asarray(raw_image, dtype=np.uint8)

    if image_numbers is None:
        image_numbers = all_indices.keys()

    images = {}
    for key, indices in all_indices.items():
        if key not in image_numbers:
            continue
        img = image[indices[0] : indices[1], indices[2] : indices[3]]
        if ensure_grayscale and len(img.shape) > 2:
            img = np.mean(img, axis=-1).astype(np.uint8)
        images[key] = img
    return images

def load_graph_images(folder, image_numbers=None, plane_numbers=None,
                      calibration_filename=None, *args, **kwargs):
    if calibration_filename is None:
        calibration_filename = folder + "/calibration_information"
    name_dict = load_dictionary(calibration_filename)["plane_names"]
    if plane_numbers is None:
        plane_numbers = np.arange(len(name_dict))
    all_images = np.zeros(len(plane_numbers), dtype=object)
    all_contents = os.listdir(folder)

    for i, plane in enumerate(plane_numbers):
        for name in all_contents:
            if name_dict[plane] in name:
                image_folder = name
                break
        images_dict = load_image_set(
            filename=folder + "/" + image_folder,
            image_numbers=image_numbers,
            calibration_filename=calibration_filename,
            *args,
            **kwargs,
        )

        all_images[i] = images_dict

    return all_images


def load_video(video_path, framewise_downsample=1, max_frames=1000):
    cap = cv2.VideoCapture(video_path)
    frames = []
    i = 0
    while cap.isOpened() and len(frames) < max_frames:
        ret, frame = cap.read()
        if not ret:
            break

        if i % framewise_downsample == 0:
            frames.append(frame)
        i = i + 1

        # this is necessary to release memory when working with long videos
        cv2.waitKey(1)
    cap.release()
    return np.array(frames)

# return a dictionary with a separate 4D numpy array for each camera
# right now this wouldn't work for MCAM videos
def load_split_video(
    video_path, calibration_filename, framewise_downsample=1, max_frames=1000
):
    full_video = load_video(video_path, framewise_downsample, max_frames=max_frames)
    all_indices = load_dictionary(calibration_filename)["crop_indices"]
    videos = {}
    for key, indices in all_indices.items():
        videos[key] = full_video[:, indices[0] : indices[1], indices[2] : indices[3]]
    return videos
