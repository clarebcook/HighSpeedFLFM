## Camera calibration 
Calibration is completed by following the prompts in `run_calibration.ipynb`. Two steps in the process require the user to manually select points from images, which is done in a GUI by running `remove_identified_vertices.py` and `select_alignment_points.py`. The notebook will indicate at what point this needs to be completed, and allow the user to use example data to skip those steps.

Depending on lighting conditions or the precise graph target being used, some settings used in the calibration notebook would need to be adjusted for successful calibration from images other than the provided example images. This would include the ``expected_spacing`` between graph vertices (cell 7) and the ``threshold_values`` set in cell 11, which determines settings used to find the approximate location of graph lines in the images. Typically, a single set of threshold values will work for all images in a calibration dataset, but there may be instances where illumination changes considerably between micro-cameras or between planes. In those cases, separate ``threshold_values`` can be set for separate images. When `threshold_values` are used to parse lines for one image, those values are saved in alongside the calibration information. If different `threshold_values` are set for different images, the values for subsequent images will default to those used in the physically closest camera. 

When selecting ``threshold_values``, the user should ensure that all graph lines in the image are identified. While effort should be made to reduce the number of falsely identified lines, having a small number of these should not significantly impact the outcome. 

The end result of this process is a single file, ``calibration_information``, which by default will be stored in the same folder as the calibration information. This is the information needed to proceed with 3D analysis. 


## Specimen alignment

run ``match_points_gui.py`` for "paint" and "alignment" points. More information will be added on this later.  