import torch
import numpy as np
from scipy import stats

# displacements should be (# points, # frames)
# this will just return the point with the average highest velocity
def get_strike_center(displacements):   
    derivs = torch.diff(displacements, axis=1)
    avg = torch.mean(torch.abs(derivs), axis=0) 
    strike_point = torch.argmax(avg) 
    return strike_point

# displacements should be (# points, # frames)
# this will return the index at which the displacements first peak
# after the strike center
def get_peak_indices(displacements):
    strike_center = get_strike_center(displacements)
    derivs = torch.diff(displacements, axis=1)
    peak_indices = torch.zeros(displacements.shape[0], dtype=torch.uint16)

    for i in range(len(displacements)):
        line_d = derivs[i, strike_center:]
        stop_point = -1
        for index, value in enumerate(line_d):
            if torch.sign(value) != torch.sign(line_d[0]):
                stop_point = index + strike_center
                break
        peak_indices[i] = stop_point
    return peak_indices

# array should be shape (# points, # values, # frames)
# this should be cropped down to the frames that are being considered
# #values would be twice the number of expected cameras
# if "treat_individually" is true, every frame in the array will be sorted separately
def sort_by_camera(array, treat_individually=False):
    # first combine the camera values
    cond_array = torch.zeros(
        (array.shape[0], int(array.shape[1] / 2), array.shape[2]), dtype=array.dtype
    )
    for i in range(cond_array.shape[1]):
        cond_array[:, i] = torch.mean(array[:, 2 * i : 2 * i + 2], axis=1)
    if treat_individually:
        cond_array, indices = torch.sort(cond_array, axis=1)
    else:
        average_values = torch.mean(cond_array, axis=-1)
        indices = torch.argsort(average_values, axis=1)
        indices = indices[:, :, None].repeat(1, 1, cond_array.shape[-1])
        cond_array = cond_array.gather(1, indices) 

    return indices, cond_array

def get_percentiles(array, value):
    array = np.sort(array) 
    index = np.where(array >= value)[0][0]
    percent = index / len(array)
    return percent

def get_random_percentile_index(array, start, stop):
    start_val = np.percentile(array, start)
    stop_val = np.percentile(array, stop) 

    indices1 = (array > start_val).astype(bool)
    indices2 = (array < stop_val).astype(bool)
    indices = indices1 & indices2
    a = np.where(indices)[0]
    np.random.shuffle(a)

    return a[0]

def convert_to_percentile(array):
    return np.vectorize(lambda x: stats.percentileofscore(array, x))(array)

# this is to bring us from a point in the world/camera frame of reference
# to a pixel in a camera
def world_frame_to_pixel(system, point, camera=2):
    # use the magnification at that z-plane to switch x, y value back to pixels 
    mag0 = system.get_magnification_at_plane(camera_number=camera, plane_mm=point[2], dim=0)
    mag1 = system.get_magnification_at_plane(camera_number=camera, plane_mm=point[2], dim=1)
    
    pixel_size_m = system.calib_manager.pixel_size
    pixels = (point[0] * mag0 / (pixel_size_m * 1e3), point[1] * mag1 / (pixel_size_m * 1e3))
    
    # then use height and calibration model to find the pixel
    # just using the reference camera to make things easier 
    ref_camera = system.reference_camera
    v0, v1 = system.get_shift_slopes(ref_camera, [pixels[0]], [pixels[1]])
    pixels = (pixels[0] - point[2] * v0, pixels[1] - point[2] * v1)
    
    return pixels