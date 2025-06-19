import numpy as np

from hsflfm.calibration.vertices_organizing_functions import (
    get_average_spacing,
    organize_by_axis
)


## Function to find incorrectly marked vertices in Calibration Images 
#   sets: array of vertices(organized by row or column) from one [plane_idx][view_idx]
#   axis: check spacing along this axis
#   spacing: avg spacing(vertically or horizontally) between points
#   max_mutiple: How many avg multiples of spacing to allow in case of points being skipped
#   gap_tolerance: Tolerance for how close a gap is to a multiple
#   decimals: Rounding for tuple
def find_irregular_points(sets, axis, spacing, max_multiple=6, gap_tolerance=0.15, decimals=4):
    irregular_points = [] # Initialize array

    # Function takes a vertex and converts to a rounded tuple
    def to_rounded_tuple(point, decimals):
        point = np.asarray(point, dtype=np.float64)
        return tuple(np.round(np.array(point), decimals=decimals))

    # Loop through each row/column in sets
    for group in sets:
        group = np.asarray(group, dtype=np.float64)
        sorted_group = group[np.argsort(group[:, axis])] # sorts each group(row/column) based on axis
        i = 0

        # Loop through sorted_group until second-to-last point to compare pairs of points
        while i < len(sorted_group) - 1:

            # Calculate distance between current point and next one along axis
            gap = sorted_group[i + 1, axis] - sorted_group[i, axis]

            # Computes how many multipes of expected spacing the gap is
                #This includes points that are in the correct position but further than one regular distance away
            ratio = gap / spacing

            # Checks if the gap is not close to multiples of spacing
            is_bad_gap = not np.any(np.isclose(ratio, np.arange(1, max_multiple + 1), atol=gap_tolerance))

            # If a gap is suspiciously large or doesn't match a multiple
            if is_bad_gap:
                # Special handling for the first gap(accounts for when first point is abnormal)
                if i == 0 and len(sorted_group) > 2:
                    next_gap = sorted_group[i + 2, axis] - sorted_group[i + 1, axis] #Checks the next gap after first point
                    next_ratio = next_gap / spacing #Calculates ratio for that next gap

                    # If the second gap looks okay, assume the first point is the irregular one and add to list, otherwise blame the second point 
                    if np.any(np.isclose(next_ratio, np.arange(1, max_multiple + 1), atol=gap_tolerance)):
                        irregular_points.append(to_rounded_tuple(sorted_group[i], decimals))
                    else:
                        irregular_points.append(to_rounded_tuple(sorted_group[i + 1], decimals))
                # If not the first gap, assume the second point in the bad pair is irregular
                else:
                    irregular_points.append(to_rounded_tuple(sorted_group[i + 1], decimals))
                i += 2  # Skip over this bad point to avoid cascading issues
            else:
                i += 1

    return irregular_points #returns list of irregular points(not dict) formatted as tuples


## Function detects irregular points in a dictionary of vertices for each plane/view.
#   vertices_dict: Dictionary of calibration vertices organized by [plane][view].
#   expected_spacing: Expected horizontal and vertical spacing.
#   max_mutiple: How many avg multiples of spacing to allow in case of points being skipped
#   gap_tolerance: Tolerance for how close a gap is to a multiple
#   decimals: Rounding for tuple
def detect_all_irregular_points(vertices_dict, expected_spacing, max_multiple=6, gap_tolerance=0.15, decimals=4):
    
    all_irregulars = {}

    for plane_num in vertices_dict:
        for view_idx in vertices_dict[plane_num]:
            raw_points = vertices_dict[plane_num][view_idx]

            # STEP 1: Filter valid points
            valid_points = [
                p for p in raw_points
                if isinstance(p, (list, tuple, np.ndarray)) and len(p) == 2 and all(isinstance(coord, (int, float)) for coord in p)
            ]
            verts = np.array(valid_points, dtype=np.float64)

            # STEP 2: Compute spacing
            h_spacing, v_spacing = get_average_spacing(verts, expected_spacing=expected_spacing)

            # STEP 3: Organize points
            rows = organize_by_axis(verts, axis=1, expected_spacing=v_spacing)
            cols = organize_by_axis(verts, axis=0, expected_spacing=h_spacing)

            # STEP 4: Find irregulars
            row_irregulars = find_irregular_points(rows, axis=0, spacing=h_spacing, max_multiple=max_multiple, gap_tolerance=gap_tolerance, decimals=decimals)
            col_irregulars = find_irregular_points(cols, axis=1, spacing=v_spacing, max_multiple=max_multiple, gap_tolerance=gap_tolerance, decimals=decimals)

            final_irregulars = list(set(row_irregulars) | set(col_irregulars))

            if final_irregulars:
                all_irregulars[(plane_num, view_idx)] = final_irregulars

    return all_irregulars