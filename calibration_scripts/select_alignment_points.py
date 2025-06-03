from hsflfm.config import home_directory
from hsflfm.calibration import CalibrationInfoManager
from hsflfm.util import load_image_set, MetadataManager

from matplotlib import pyplot as plt
from PIL import Image

# select any specimen filmed under the calibration conditions
# alternatively, the calibration folder and filename can be manually specified
specimen = "20240506_OB_6" # New change: redefined specimen
mm = MetadataManager(specimen)
#calibration_filename = r"C:/Users/abhin/Documents/Graduate School/Patek Research Docs/test_calibration.json"
calibration_filename = mm.calibration_filename
calibration_folder = mm.calibration_folder
calibration_manager = CalibrationInfoManager(calibration_filename)
image_filename = mm.light_calibration_filename

image_numbers = calibration_manager.image_numbers

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
ax0.imshow(im, cmap="gray")
image = get_current_image()
im = ax.imshow(image)
fig.suptitle(get_title())
cid = fig.canvas.mpl_connect("button_press_event", select_point)

plt.show()

calibration_manager.approx_alignment_points = points_dict
calibration_manager.save_all_info()
