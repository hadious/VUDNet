import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from PIL import Image
from sklearn.decomposition import PCA




def calculate_centroid(matrix):
        return np.mean(matrix, axis=0)
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
    
def rotate_matrix_3d(matrix, angle_x, angle_y, angle_z, centroid):
    matrix_np = np.array(matrix)    
    translated_matrix = matrix_np - centroid
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
    
    R_combined = Rz @ Ry @ Rx
    rotated_matrix = np.matmul(translated_matrix, R_combined.T)
    rotated_translated_matrix = rotated_matrix + centroid
    
    return rotated_translated_matrix

matrix = np.load('VAE/VAE_DEPTH/PDVAE2d_obj_Plane.002_view4.png.npy')


x = np.arange(matrix.shape[1])
y = np.arange(matrix.shape[0])
X, Y = np.meshgrid(x, y)

fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')

ax.plot_surface(X, Y, matrix, cmap='viridis')

ax.set_xlabel('X axis')
ax.set_ylabel('Y axis')
ax.set_zlabel('Z axis')
ax.set_title('3D Surface Plot')

plt.show()

npz_file = np.load('U-Net/parabolic_concave/depth_obj_Plane.002_view4.npz')
depthMap = npz_file['arr_0'] 

depthMap = np.fliplr(depthMap)

depth_map_img = Image.fromarray(depthMap)
depth_map_cropped_img = depth_map_img.crop((500, 50, 1400, 900))
depth_map_resized_img = depth_map_cropped_img.resize((240, 320))
depth_map_resized_np = np.array(depth_map_resized_img)

depth_map = depth_map_resized_np

height, width = depth_map.shape
depth_matrix = []

for y in range(height):
    for x in range(width):
        depth_value = depth_map[y, x]
        depth_matrix.append([x, y, depth_value])

depth_matrix = np.array(depth_matrix)

centroid = calculate_centroid(depth_matrix)

pca = PCA(n_components=3)
pca.fit(depth_matrix)
normal_vector = pca.components_[2]  

yaw, pitch, roll = calculate_rotation_angles(normal_vector)
yaw = 0
pitch = 0
roll = roll if roll > 0 else roll + 180 
rotated_matrix = rotate_matrix_3d(depth_matrix, roll, pitch, yaw, centroid)
rotated_matrix = rotate_matrix_3d(rotated_matrix, 0, 0, 180, centroid)

third_column = rotated_matrix[:, 2]
third_column = third_column * 10
third_column = third_column **2
rotated_matrix[:, 2] = third_column

x, y, values = zip(*rotated_matrix)
min_x, min_y = min(x), min(y)
x = np.array(x) - min_x
y = np.array(y) - min_y

width = int (max(x) + 1)
height = int (max(y) + 1)

array_2d = np.zeros((height, width))  

for i, (x_i, y_i, value) in enumerate(zip(x, y, values)):
    array_2d[int(y_i), int(x_i)] = value  

normalized_depthmap = (array_2d - np.min(array_2d)) / np.ptp(array_2d)


x = np.arange(normalized_depthmap.shape[1])
y = np.arange(normalized_depthmap.shape[0])
X, Y = np.meshgrid(x, y)

fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')

ax.plot_surface(X, Y, normalized_depthmap, cmap='viridis')

ax.set_xlabel('X axis')
ax.set_ylabel('Y axis')
ax.set_zlabel('Z axis')
ax.set_title('3D Surface Plot')

plt.show()


import numpy as np
import matplotlib.pyplot as plt

matrix_npy = np.load('VAE/VAE_DEPTH/PDVAE2d_obj_Plane.002_view4.png.npy')
        
depthMap = depth_map_resized_np


difference = matrix_npy - normalized_depthmap

plt.figure(figsize=(10, 8))
plt.imshow(difference, cmap='coolwarm', interpolation='nearest')
plt.colorbar()
plt.title('Heatmap of Matrix Difference')
plt.xlabel('X axis')
plt.ylabel('Y axis')
plt.show()


