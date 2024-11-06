from hsflfm.config import home_directory
from hsflfm.calibration import CalibrationInfoManager
from hsflfm.util import load_image_set

from matplotlib import pyplot as plt
from PIL import Image

calibration_folder = home_directory + "/Videos/20240427/20240427_calibration"
calibration_filename = "delete_calibration_information"  # calibration_folder + '/calibration_information'
calibration_manager = CalibrationInfoManager(calibration_filename)

image_numbers = calibration_manager.image_numbers
# frequently easiest to use an ant instead of the graph
# could clean this up to just automatically select one of the ant images
image_filename = home_directory + '/Videos/20240427/OB_5/20240427_OB_5_alignment_ring_0.tiff' 

points_dict = {}

global camera_number
camera_number = 0
calibration_manager = CalibrationInfoManager(calibration_filename)


def get_title():
    return f"Double click to select alignment point for camera {camera_number}"


def get_current_image():
    image = load_image_set(
        filename=image_filename,
        calibration_filename=calibration_filename,
        image_numbers=[camera_number],
    )[camera_number]
    return image


def select_point(event):
    if not event.dblclick:
        return

    global camera_number

    ix, iy = event.xdata, event.ydata
    points_dict[camera_number] = (int(iy), int(ix))

    camera_number = camera_number + 1

    if camera_number > image_numbers[-1]:
        plt.close()
        return

    image = get_current_image()
    im.set_data(image)
    fig.suptitle(get_title())
    fig.canvas.draw()


fig, (ax0, ax) = plt.subplots(1, 2)
im = Image.open(image_filename)
ax0.imshow(im, cmap='gray')
image = get_current_image()
im = ax.imshow(image)
fig.suptitle(get_title())
cid = fig.canvas.mpl_connect("button_press_event", select_point)

plt.show()

calibration_manager.approx_alignment_points = points_dict
calibration_manager.save_all_info()
