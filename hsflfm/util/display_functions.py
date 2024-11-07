import cv2
from matplotlib import pyplot as plt
import numpy as np
import time
from matplotlib.patches import Circle
from IPython.display import display, clear_output


## functions for use during calibration, to look at images with calibrated lines and points
# since this is for display purposes
# x and y length should be in display coordinates
def _get_line_coords(r, theta, xlength, ylength):
    # Stores the value of cos(theta) in a
    a = np.cos(theta)

    # Stores the value of sin(theta) in b
    b = np.sin(theta)

    # x0 stores the value rcos(theta)
    x0 = a * r

    # y0 stores the value rsin(theta)
    y0 = b * r

    # x1 stores the rounded off value of (rcos(theta)-1000sin(theta))
    x1 = int(x0 + xlength * (-b))

    # y1 stores the rounded off value of (rsin(theta)+1000cos(theta))
    y1 = int(y0 + ylength * (a))

    # x2 stores the rounded off value of (rcos(theta)+1000sin(theta))
    x2 = int(x0 - xlength * (-b))

    # y2 stores the rounded off value of (rsin(theta)-1000cos(theta))
    y2 = int(y0 - ylength * (a))
    return x1, y1, x2, y2


# The below for loop runs till r and theta values
# are in the range of the 2d array
def display_with_lines(
    image,
    lines,
    xlength=10000,
    ylength=10000,
    title="Image with Lines",
    display_downsample=1,
):
    new_image = image.copy()
    for r_theta in lines:
        arr = np.array(r_theta, dtype=np.float64)
        r, theta = arr
        x1, y1, x2, y2 = _get_line_coords(r, theta, xlength, ylength)

        # cv2.line draws a line in img from the point(x1,y1) to (x2,y2).
        # (0,0,255) denotes the colour of the line to be
        # drawn. In this case, it is red.
        cv2.line(new_image, (x1, y1), (x2, y2), (0, 0, 255), 2 * display_downsample)
    plt.figure()
    plt.title(title)
    plt.imshow(new_image[::display_downsample, ::display_downsample])


def display_with_points(
    image, points, title="Image with Points", display_downsample=1, *args, **kwargs
):
    plt.figure()
    image = image[::display_downsample, ::display_downsample]
    plt.imshow(image, *args, **kwargs)
    for point in points:
        circ = Circle(
            (point[0] / display_downsample, point[1] / display_downsample),
            radius=4,
            color="red",
        )
        plt.gca().add_artist(circ)
    plt.title(title)

    return image


def play_video(frames_array, fps=30, *args, **kwargs):
    """
    Display a 4D NumPy array (video) as a video with a given frames per second (fps).
    """
    num_frames = frames_array.shape[0]

    # Iterate over frames and display them
    _ = plt.figure()
    canvas = plt.imshow(frames_array[0], *args, **kwargs)
    plt.clim(np.min(frames_array), np.max(frames_array))
    # plt.colorbar()
    plt.axis("off")
    for i in range(num_frames):
        start_time = time.time()
        canvas.set_data(frames_array[i])
        display(plt.gcf())
        clear_output(wait=True)
        passed_time = time.time() - start_time
        remaining_time = 1 / fps - passed_time
        time.sleep(max(remaining_time, 0))  # Control playback speed
