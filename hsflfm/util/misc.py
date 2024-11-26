from datetime import datetime
import numpy as np
import scipy.signal


def get_timestamp():
    str_format = "%Y%m%d_%H%M%S"
    time_string = datetime.now().strftime(str_format)
    return time_string


def cross_image(im1_gray, im2_gray):
    # get rid of the averages, otherwise the results are not good
    im1_gray -= np.mean(im1_gray)
    im2_gray -= np.mean(im2_gray)

    # calculate the correlation image; note the flipping of onw of the images
    corr_image = scipy.signal.fftconvolve(im1_gray, im2_gray[::-1, ::-1], mode="same")
    unraveled = np.unravel_index(np.argmax(corr_image), corr_image.shape)
    middle = corr_image.shape[0] / 2, corr_image.shape[1] / 2
    return unraveled[0] - middle[0], unraveled[1] - middle[1]
