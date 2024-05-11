import numpy as np 
import matplotlib.pyplot as plt
from PIL import Image 
from scipy.spatial.transform import Rotation as R
from sklearn.decomposition import PCA

def calculate_rotation_angles(normal_vector):
    azimuthal_angle = np.arctan2(normal_vector[1], normal_vector[0])  
    polar_angle = np.arccos(normal_vector[2] / np.linalg.norm(normal_vector))  

    return np.degrees(azimuthal_angle), np.degrees(polar_angle)


def rotate_matrix_3d(matrix, angle_x, angle_y, angle_z):

    matrix_np = np.array(matrix)
    roll = np.array([[1, 0, 0],
                           [0, np.cos(np.radians(angle_x)), -np.sin(np.radians(angle_x))],
                           [0, np.sin(np.radians(angle_x)), np.cos(np.radians(angle_x))]])

    pitch = np.array([[np.cos(np.radians(angle_y)), 0, np.sin(np.radians(angle_y))],
                           [0, 1, 0],
                           [-np.sin(np.radians(angle_y)), 0, np.cos(np.radians(angle_y))]])

    yaw = np.array([[np.cos(np.radians(angle_z)), -np.sin(np.radians(angle_z)), 0],
                           [np.sin(np.radians(angle_z)), np.cos(np.radians(angle_z)), 0],
                           [0, 0, 1]])

    rotation_matrix = np.matmul(yaw, np.matmul(pitch, roll))
    rotated_matrix_np = np.matmul(rotation_matrix,matrix_np.T).T
    return rotated_matrix_np



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
        ax.set_title('Depth Map Visualization')
        plt.show()
depth_path = "2D_fringe/depth_obj_Sphere_view1.npz"

depth_data = np.load(depth_path)
depth_map = depth_data['arr_0'] 


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

# mean_values = np.mean(depth_matrix, axis=0)
# depth_matrix[:, :2] -= mean_values[:2]


plot_3d_matrix(depth_matrix)




pca = PCA(n_components=3)
pca.fit(depth_matrix)
normal_vector = pca.components_[2] 


azimuthal_angle, polar_angle = calculate_rotation_angles(normal_vector)

print("Normal Vector:", normal_vector)
print("Azimuthal Angle (around Z-axis):", azimuthal_angle)
print("Polar Angle (around X or Y axis):", polar_angle)


angle_x =  polar_angle
angle_z = azimuthal_angle
angle_y = 0



rotated_matrix = rotate_matrix_3d(depth_matrix, angle_x, angle_y, angle_z)

 
third_column = rotated_matrix[:, 2]
third_column = third_column * 10
# third_column = third_column **2
rotated_matrix[:, 2] = third_column



plot_3d_matrix(rotated_matrix)

x, y, values = zip(*rotated_matrix)

x = np.floor(x).astype(int)
y = np.floor(y).astype(int)

max_x = max(x) + 1
max_y = max(y) + 1

print (max_x,max_y,min(x), min(y))


max_x = int(max_x)
max_y = int(max_y)


array_2d = np.zeros((max_x, max_y))
array_2d[x, y] = values


plot_depth_map_3d(array_2d)