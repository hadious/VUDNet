from PIL import Image
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


def plot_3d_matrix(matrix):
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    
    x = matrix[:, 0]
    y = matrix[:, 1]
    z = matrix[:, 2]
    
    colormap = plt.cm.viridis  
    normalize = plt.Normalize(vmin=min(z), vmax=max(z))
    colors = colormap(normalize(z))

    ax.scatter(x, y, z, c=colors, marker='.')
    
    ax.set_xlabel('X Label')
    ax.set_ylabel('Y Label')
    ax.set_zlabel('Z Label')
    
    scalar_map = plt.cm.ScalarMappable(norm=normalize, cmap=colormap)
    scalar_map.set_array(z)
    plt.colorbar(scalar_map, ax=ax, label='Depth')

    plt.show()



def plot_depth_map_3d(depth_map):
        X, Y = np.meshgrid(np.arange(depth_map.shape[1]), np.arange(depth_map.shape[0]))

        # Plot the depth map in 3D
        fig = plt.figure()
        ax = fig.add_subplot(111, projection='3d')
        ax.plot_surface(X, Y, depth_map, cmap='viridis')  # Use 'viridis' colormap for better perception of depth
        ax.set_xlabel('Width')
        ax.set_ylabel('Height')
        ax.set_zlabel('Depth')
        ax.set_title('3D Depth Map Visualization')
        plt.show()


depth_path = "2D_fringe/depth_obj_Sphere_view2.npz"
# depth_path = "depth_obj_Plane.002_view0.npz"
depth_data = np.load(depth_path)
depth_map = depth_data['arr_0'] 

depth_map = np.fliplr(depth_map)



depth_map_img = Image.fromarray(depth_map)
depth_map_cropped_img = depth_map_img.crop((480, 50, 1400, 900))
depth_map_resized_img = depth_map_cropped_img.resize((240, 320))
depth_map_resized_np = np.array(depth_map_resized_img)

depth_map = depth_map_resized_np


plot_depth_map_3d(depth_map)


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
print (roll)
# angle_x = -0.14
# angle_y = 0
# angle_z = 0
# rotated_matrix = rotate_matrix_3d(depth_matrix, angle_x, angle_y, angle_z, centroid)
yaw = 0
pitch = 0
roll = roll if roll > 0 else roll + 180 
rotated_matrix = rotate_matrix_3d(depth_matrix, roll, pitch, yaw, centroid)

plot_3d_matrix(rotated_matrix)

rotated_matrix = rotate_matrix_3d(rotated_matrix, 0, 0, 180, centroid)


third_column = rotated_matrix[:, 2]
third_column = third_column * 10
third_column = third_column **2
rotated_matrix[:, 2] = third_column


plot_3d_matrix(rotated_matrix)



x, y, values = zip(*rotated_matrix)
min_x, min_y = min(x), min(y)
x = np.array(x) - min_x
y = np.array(y) - min_y

width = int (max(x) + 1)
height = int (max(y) + 1)

array_2d = np.zeros((height, width))  

for i, (x_i, y_i, value) in enumerate(zip(x, y, values)):
    array_2d[int(y_i), int(x_i)] = value  
    


plot_depth_map_3d(array_2d)

