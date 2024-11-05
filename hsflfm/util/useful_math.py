import numpy as np
from scipy.linalg import lstsq

def rot_trans_from_matrix(matrix):
    return 

def matrix_from_rot_trans(x, y, z, roll, pitch, yaw):
    return 

# the returned matrix assumes X points are scaled then transformed
# to give the "transformed_points"
def procrustes_analysis(X, Y, allow_scale=True):
    return matrix, scale, transformed_points

# this returns coefficients defining the plane [a, b, c, d]
# where ax + by + cz + d = 0
def estimate_plane(points):
    # Convert points to NumPy array for easier manipulation
    points = np.array(points)
    
    # Extract x, y, and z coordinates
    x, y, z = points[:, 0], points[:, 1], points[:, 2]
    
    # Create the design matrix
    A = np.column_stack((x, y, np.ones_like(x)))
    
    # Solve the least squares problem to find coefficients [a, b, c, d] for the plane equation ax + by + cz + d = 0
    coeffs, _, _, _ = lstsq(A, -z)
    
    coeffs =  np.asarray([coeffs[0], coeffs[1], 1, coeffs[2]]) 
    coeffs = coeffs / np.linalg.norm(coeffs[:3]) 
    return coeffs

# https://stackoverflow.com/questions/9605556/how-to-project-a-point-onto-a-plane-in-3d
def project_point_on_plane(point, plane_coeffs):
    # first make sure coefficients are normalized
    plane_coeffs = plane_coeffs / np.linalg.norm(plane_coeffs[:3])
    unit_norm = plane_coeffs[:3]
    
    # then get the distance from the point to the plane,
    # with dot product of vector to point and the normal 
    distance = np.dot(unit_norm, point) + plane_coeffs[3] 
    difference = unit_norm * distance 
    
    plane_point = point - difference
    
    # confirm that it's really on the plane 
    summed = (plane_point[0] * plane_coeffs[0] + plane_point[1] * plane_coeffs[1] +
              plane_point[2] * plane_coeffs[2] + plane_coeffs[3])

    assert abs(summed) < 1e-10
        
    return plane_point

# convenience function to multipy points of shape [N, 3] with a 4x4 matrix 
def matmul(M, points):
    points = np.asarray(points)
    points = np.concatenate((points, np.ones((points.shape[0], 1))), axis=1) 
    transformed_points = np.linalg.matmul(M, points.T).T[:, :3]
    return transformed_points