# Post-Calibration Scripts

This folder contains GUI tools for managing post-calibration point identification, alignment, and transfer across multiple ant strike recordings.

> **Note:** Make sure the system calibration workflow is completed before using these scripts.

---

## Contents

- `match_points_gui.py`: Annotate initial paint or alignment points, to match points between the perspective images.
- `manual_strike_transfer.py`: GUI to transfer and verify points across strikes, correct misaligned points.
- `manual_alignment_gui.py`: 3D viewer for global alignment with ant exoskeleton.
- `process_from_pre_alignment.ipynb`: Notebook to run 3D analysis following alignment to ant coordinate frame. 
- `process_without_alignment.ipynb`: Notebook to run 3D analysis for any sample, without aligning to ant coordinate frame. This still requires matching points between perspective images.

---

## Suggested Workflow

For ant videos:
1. Run `match_points_gui.py` to collect alignment and paint points.
2. Run `manual_strike_transfer.py` for the specimen.
2a. From this GUI, launch the manual alignment GUI to ensure matched points are correctly aligned to the 3D ant model. 
2b. Use the GUI to:
   - Use “Add Missing Points” for unmatched points
   - Use “Remove Points” for incorrect or mistakenly placed points
   - Use “Manual Align” tools as needed
   - Open the 3D viewer to view alignment

3. Open ``process_from_pre_alignment.ipynb``. Set where to save results and run script. 




For arbitrary sample without aligning to ant coordinate frame: 
1. Run `match_points_gui.py` to match points between views.
2. Modify filenames and run `process_without_alignment.ipynb`


## Usage Guide

### 1. Annotate Initial Points

Run the following twice per specimen to manually label points for alignment and painted markers (points to track) in the first video for the specimen:

```bash
python match_points_gui.py --specimen_number 20240506_OB_6 --point_type alignment --exclude_prior_points --demo 
```

- The ``specimen_number`` should be a unique identifier included in the dataset csv sheet (from the ``Alignment Tag`` column)- defaults to ``20240506_OB_6``
- ``point_type`` should be ``alignment`` or ``paint``. This specifies whether the selected points are the fiducial points on the ant head used for coarse alignment, or the points to be tracked during SDV. Points are saved automatically per camera.
- If ``exclude_prior_points`` flag is included, previously saved points will not be loaded and will be over-written if not in demo mode. 
- If ``demo`` flag is included, points will not be saved. 
- Run **twice**: once for `"alignment"`, and once for `"paint"` to align with ant coordinate frame. To proceed with 3D analysis without alignment to ant coordinate frame, only run once for `"paint"`. 
- By default, points are saved alongside the alignment images (see description of file organization in the main folder README). 

---

### 2. Transfer and Verify Points
This is used to ensure points are correctly matched between multiple videos of the same specimen. 

Run:

```bash
python manual_strike_transfer.py
```

This GUI transfers points from the first strike to all other strikes for a specimen.

Configure inputs near the bottom of the file (around line 868):

```python
specimen_numbers = ["20240506_OB_6"]
save_folder = "path/to/output/folder"
demo_mode = False  # Set to True to disable file saving
```

- A PyQt GUI will launch. Alignment results will be saved in the specified ``save_folder``, which can be reloaded in the next step. 
---

2a. Launch Manual Alignment Viewer

To confirm the 3D points are correctly aligned with the 3D model of the ant exoskeleton:

- Click **“Open Manual Alignment Viewer”** in the GUI.
- This launches `manual_alignment_gui.py`, which opens a browser-based 3D Dash viewer. If this does not open automatically, check the terminal where the GUI was launched - there should be an address, which copied to a browser manually. This may take several seconds.
- Use the sliders to adjust the alignment of the dynamic points so that they lie on the surface of the exskeleton, once finished click 'done' to export. Then, in the strike transfer GUI, click `Load alignment values` to use these values. 

This should be completed for the first strike video for the specimen

---

2b. Ensure points are transferred correctly between videos 

Use the buttons described below to fix any mistakes with transferring points between videos. When done with each strike, click **Approve** to move onto the next strike video. 

## GUI Button Reference

The manual_strike_transfer GUI includes several buttons, each corresponding to a specific mode or action:

- **Add Missing Points** — Switches to “add points” mode. Prompts users to click missing points in the current strike.
- **Skip Point** — Skips the current point if it cannot be reliably identified. Can be used in “add points” or “manual align” modes.
- **View All** — Switches to “view all” mode. Displays all points on all camera views.
- **Remove Points** — Switches to “remove points” mode. Click any point to remove it from the current strike.
- **Start New Manual Transfer** — Switches to “manual align” mode. Manually match points from the previous strike to the current strike.
- **Continue Manual Transfer** — Resume a previous manual alignment session to add more points.
- **Re-run Alignment** — Recalculates alignment based on current manually added points.
- **Approve** — Finalizes the current strike and advances to the next.
- **Open Manual Alignment Viewer** —  Launches the browser-based 3D model viewer from `manual_alignment_gui.py` as a subprocess.
- **Load Alignment Values** - Loads user adjusted Alignment Values exported from the 3D model viewer.

