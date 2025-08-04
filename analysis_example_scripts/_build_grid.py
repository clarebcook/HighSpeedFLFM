# this script holds functions to build a grid over the ant coordinate frame
# that is used in several of the analysis scripts 

import numpy as np 
from matplotlib import pyplot as plt 

# establish grid for comparing regions 
# form the grid
avg_ant_scale = 1693 
xmin = -1500 
xmax = 1200 
ymin = -600
ymax = 700

# load the outline for display
outline_filename = "../hsflfm/ant_model/model_outline.npy"
outline = np.load(outline_filename)


def build_grid(y_count=7, analyzer=None, show=True,
               show_numbers=True):
    y_count = 7
    x_count = int(y_count * (xmax - xmin) / (ymax - ymin))

    y_bounds = np.linspace(ymin, ymax + 1, y_count + 1) 
    x_bounds = np.linspace(xmin, xmax + 1, x_count + 1) 

    if analyzer is not None and show:
        locs = np.asarray(analyzer.all_results["start_locations_std"])[:, :2]
        jitter = 80 
        locs_jittered = locs + np.random.random(locs.shape) * jitter - jitter / 2
        plt.scatter(locs_jittered[:, 1] / avg_ant_scale, locs_jittered[:, 0] / avg_ant_scale, s=0.8, rasterized=True)
        ax = plt.gca()
        ax.set_aspect("equal")

        for i, yb in enumerate(y_bounds):
            plt.plot([yb / avg_ant_scale, yb / avg_ant_scale],
                    [np.min(x_bounds) / avg_ant_scale, np.max(x_bounds) / avg_ant_scale],
                    color='black')
            if i < len(y_bounds) - 1 and show_numbers:
                plt.text(y=np.max(x_bounds) / avg_ant_scale + 0.01,
                        x=(yb + y_bounds[i + 1]) / (avg_ant_scale * 2) - 0.04,
                        s=int(i), color='red')
        for i, xb in enumerate(x_bounds):
            plt.plot([np.min(y_bounds) / avg_ant_scale, np.max(y_bounds) / avg_ant_scale],
                    [xb / avg_ant_scale, xb / avg_ant_scale],
                    color='black')
            if i < len(x_bounds) - 1 and show_numbers:
                plt.text(x=np.min(y_bounds) / avg_ant_scale - 0.12,
                        y=(xb + x_bounds[i + 1]) / (avg_ant_scale * 2) - 0.03,
                        s=int(i), color='red')
            
        plt.plot(outline[:, 1] / avg_ant_scale, outline[:, 0] / avg_ant_scale,
                color='black')

        plt.xlabel("x (mm)")
        plt.ylabel("y (mm)") 

    return x_bounds, y_bounds 