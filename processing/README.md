# Post-Calibration Scripts

This folder contains GUI tools for managing post-calibration point identification, alignment, and transfer across multiple ant strike recordings.

> **Note:** Make sure the system calibration workflow is completed before using these scripts.

---

## Contents

- `match_points_gui.py`: Annotate initial paint or alignment points.
- `manual_strike_transfer.py`: GUI to transfer and verify points across strikes, correct misaligned points.
- `manual_alignment_gui.py`: 3D viewer for global alignment with ant exoskeleton.

---

## Usage Guide

### 1. Annotate Initial Points

Run the following once per specimen to manually label points for alignment or painted markers:

```bash
python match_points_gui.py
```

- Toggle point mode around line 25:
  ```python
  point_type = "alignment"  # or "paint"
  ```
- Points are saved automatically per camera.
- Run **twice**: once for `"alignment"`, and once for `"paint"`.

---

### 2. Transfer and Verify Points

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

- A PyQt GUI will launch.
---

### 3. Launch Manual Alignment Viewer

If the 3D point alignment looks incorrect:

- Click **“Open Manual Alignment Viewer”** in the GUI.
- This launches `manual_alignment_gui.py`, which opens a browser-based 3D Dash viewer.

---

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
- **Open Manual Alignment Viewer** —  Launches the browser-based 3D model viewer.

---

## Suggested Workflow

1. Run `match_points_gui.py` to collect alignment and paint points.
2. Run `manual_strike_transfer.py` for the specimen.
3. Use the GUI to:
   - Use “Add Missing Points” for unmatched points
   - Use “Remove Points” for incorrect or mistakenly placed points
   - Use “Manual Align” tools as needed
   - Open the 3D viewer if alignment looks incorrect
4. Use the 3D viewer if alignment of points needs to be adjusted.(Add more here when functionality with the coords is added)
5. Once you're satisfied with the strike, click **“Approve”**. The GUI will automatically close once all strikes have been processed.
