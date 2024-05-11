import numpy as np
import matplotlib.pyplot as plt
from scipy.spatial.transform import Rotation as R
from sklearn.decomposition import PCA

def calculate_rotation_angles(normal_vector):
    normal_vector = normal_vector / np.linalg.norm(normal_vector)
    ground_normal = np.array([0, 0, 1])
    
    v = np.cross(normal_vector, ground_normal)
    s = np.linalg.norm(v)
    c = np.dot(normal_vector, ground_normal)
    skew_symmetric_matrix = np.array([[0, -v[2], v[1]], [v[2], 0, -v[0]], [-v[1], v[0], 0]])
    rotation_matrix = np.eye(3) + skew_symmetric_matrix + np.dot(skew_symmetric_matrix, skew_symmetric_matrix) * ((1 - c) / (s ** 2))
    
    yaw = np.arctan2(rotation_matrix[1, 0], rotation_matrix[0, 0])
    pitch = np.arcsin(-rotation_matrix[2, 0])
    roll = np.arctan2(rotation_matrix[2, 1], rotation_matrix[2, 2])

    return np.degrees(yaw), np.degrees(pitch), np.degrees(roll)

def calculate_centroid(matrix):
    return np.mean(matrix, axis=0)

def rotate_matrix_3d(matrix, angle_x, angle_y, angle_z, centroid):
    matrix_np = np.array(matrix) - centroid  # Translate to origin (centroid)
    
    # Define rotation matrices
    Rx = np.array([
        [1, 0, 0],
        [0, np.cos(np.radians(angle_x)), -np.sin(np.radians(angle_x))],
        [0, np.sin(np.radians(angle_x)), np.cos(np.radians(angle_x))]
    ])
    
    Ry = np.array([
        [np.cos(np.radians(angle_y)), 0, np.sin(np.radians(angle_y))],
        [0, 1, 0],
        [-np.sin(np.radians(angle_y)), 0, np.cos(np.radians(angle_y))]
    ])
    
    Rz = np.array([
        [np.cos(np.radians(angle_z)), -np.sin(np.radians(angle_z)), 0],
        [np.sin(np.radians(angle_z)), np.cos(np.radians(angle_z)), 0],
        [0, 0, 1]
    ])
    
    # Combined rotation matrix, R = Rz * Ry * Rx
    R = np.dot(Rz, np.dot(Ry, Rx))
    
    # Apply rotation
    rotated_matrix_np = np.dot(matrix_np, R.T) + centroid  # Translate back after rotation
    
    return rotated_matrix_np

# Load your depth map and convert it to a point cloud as per your existing code
# depth_map = ...

# Assuming depth_matrix is ready and contains your point cloud as [x, y, depth]
# depth_matrix = ...

# Calculate the centroid of the point cloud
centroid = calculate_centroid(depth_matrix)

# Fit PCA to find the normal vector of the point cloud
pca = PCA(n_components=3)
pca.fit(depth_matrix)
normal_vector = pca.components_[2]  # Assuming the normal vector is the last component

# Calculate rotation angles to align with the XY plane
yaw, pitch, roll = calculate_rotation_angles(normal_vector)

# Rotate the point cloud to align it
rotated_matrix = rotate_matrix_3d(depth_matrix, roll, pitch, yaw, centroid)

# Now, you can visualize your rotated_matrix with plot_3d_matrix or any visualization function you prefer.
