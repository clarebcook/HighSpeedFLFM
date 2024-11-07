import numpy as np
import torch
from scipy.linalg import lstsq

# from chatGPT
def rotation_matrix_to_euler(rotation_matrix):
    """
    Convert a 3x3 rotation matrix to Euler angles (yaw, pitch, roll) in the ZYX convention.
    
    Args:
        rotation_matrix (array): 3x3 rotation matrix.
    
    Returns:
        euler_angles (array): Euler angles (yaw, pitch, roll) in radians.
    """
    # Extract individual elements for clarity
    r11, r12, r13 = rotation_matrix[0]
    r21, r22, r23 = rotation_matrix[1]
    r31, r32, r33 = rotation_matrix[2]

    # Compute yaw (around z-axis)
    yaw = np.arctan2(r21, r11)

    # Compute pitch (around y-axis)
    pitch = np.arctan2(-r31, np.sqrt(r32**2 + r33**2))

    # Compute roll (around x-axis)
    roll = np.arctan2(r32, r33)

    # Return Euler angles in the order (yaw, pitch, roll)
    return yaw, pitch, roll

def rot_trans_from_matrix(matrix):
    rotation_matrix = matrix[:3, :3]
    translation_vector = matrix[:3, 3]

    yaw, pitch, roll = rotation_matrix_to_euler(rotation_matrix)
    return translation_vector[0], translation_vector[1], translation_vector[2], roll, pitch, yaw

def euler_to_rotation_matrix(roll_rad, pitch_rad, yaw_rad):
    # Rotation matrices around x, y, z axes
    R_x = np.array(
        [
            [1, 0, 0],
            [0, np.cos(roll_rad), -np.sin(roll_rad)],
            [0, np.sin(roll_rad), np.cos(roll_rad)],
        ]
    )

    R_y = np.array(
        [
            [np.cos(pitch_rad), 0, np.sin(pitch_rad)],
            [0, 1, 0],
            [-np.sin(pitch_rad), 0, np.cos(pitch_rad)],
        ]
    )

    R_z = np.array(
        [
            [np.cos(yaw_rad), -np.sin(yaw_rad), 0],
            [np.sin(yaw_rad), np.cos(yaw_rad), 0],
            [0, 0, 1],
        ]
    )

    # Combined rotation matrix (Z * Y * X)
    R = R_z @ R_y @ R_x
    return R

def matrix_from_rot_trans(x, y, z, roll, pitch, yaw):
    # Create a 4x4 transformation matrix
    T = np.eye(4)

    # Set the translation part
    T[0, 3] = x
    T[1, 3] = y
    T[2, 3] = z

    # Get the rotation matrix
    R = euler_to_rotation_matrix(roll, pitch, yaw)

    # Set the rotation part in the transformation matrix
    T[0:3, 0:3] = R

    return T 

# the returned matrix assumes X points are scaled then transformed
# to give the "transformed_points"
def procrustes_analysis(X, Y, allow_scale=True):
    # Center the points to remove translation
    X_mean = np.mean(X, axis=0)
    Y_mean = np.mean(Y, axis=0)
    X_centered = X - X_mean
    Y_centered = Y - Y_mean
    
    # Compute the cross-covariance matrix
    cov_matrix = np.dot(X_centered.T, Y_centered)
    
    # Perform Singular Value Decomposition (SVD) on the covariance matrix
    U, _, Vt = np.linalg.svd(cov_matrix)
    
    # Compute the rotation matrix
    rotation_matrix = np.dot(U, Vt)
    
    # Compute the scaling factor
    if allow_scale:
        scale = np.trace(np.dot(rotation_matrix.T, cov_matrix)) / np.trace(np.dot(X_centered.T, X_centered))
    else:
        scale = 1
    
    # Apply the rotation and scaling to one set of points
    transformed_points = scale * np.dot(X_centered, rotation_matrix) + Y_mean
    
    # Compute the translation
    translation_vector = Y_mean - np.dot(X_mean, rotation_matrix) * scale

    matrix = np.zeros((4, 4))
    matrix[:3, :3] = rotation_matrix.T 
    matrix[:3, -1] = translation_vector / scale
    matrix[-1, -1] = 1
                                         
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

def torch_matmul(M, points):
    points = torch.asarray(points) 
    points = torch.concatenate((points, torch.ones((points.shape[0], 1))), axis=1)
    transformed_points = torch.matmul(M, points.T).T[:, :3]
    return transformed_points

# convenience function to multipy points of shape [N, 3] with a 4x4 matrix 
def matmul(M, points):
    if isinstance(M, torch.Tensor):
        return torch_matmul(M, points)

    points = np.asarray(points)
    points = np.concatenate((points, np.ones((points.shape[0], 1))), axis=1) 
    transformed_points = np.linalg.matmul(M, points.T).T[:, :3]
    return transformed_points
