from keras import backend as Kend
import tensorflow as tf
import numpy as np
import h5py
import matplotlib.pyplot as plt
from keras.models import load_model
from skimage.restoration import unwrap_phase
from scipy.ndimage import sobel, convolve
from PIL import Image
from sklearn.decomposition import PCA


img_height = 320
img_width = 240


Kend.clear_session()
model = load_model("./U-Net/DYnet++.h5", custom_objects={"tf": tf})

dir_base = './IIIC/'
test_dir = dir_base +'Ic_'

# Calculate the root-mean-square error
def rms(X):
    square = np.square(X)
    mse = square.mean()
    rmse = np.sqrt(mse)
    return rmse

# Display the phase difference
def test_phase_difference(img):
    
    test_x = np.empty((1, img_height, img_width, 1), dtype="float32")
    img = img.crop((500, 50, 1400, 900))
    img_resized = img.resize((img_width, img_height))
    img_array = np.array(img_resized, dtype=np.float32)
    img_normalized = img_array / 255.0
    test_x[0, :, :, 0] = img_normalized
    
    
  
    #Model predictions
    (prediction_M, prediction_D) = model.predict(test_x)
    Mx = prediction_M[0,:,:,0]
    Dx = prediction_D[0,:,:,0]
    My = prediction_M[0,:,:,1]
    Dy = prediction_D[0,:,:,1]
    dis_integer_mx = Mx.reshape((img_height, img_width)) 
    dis_integer_my = My.reshape((img_height, img_width)) 
    dis_integer_dx = Dx.reshape((img_height, img_width)) 
    dis_integer_dy = Dy.reshape((img_height, img_width)) 
    
    wpx = np.arctan2(dis_integer_mx,dis_integer_dx)
    wpy = np.arctan2(dis_integer_my,dis_integer_dy)
    
    uwpx = unwrap_phase(wpx)
    uwpy = unwrap_phase(wpy)
    
    fx = 1
    fy = 1
    slope_x = (1 / (2 * np.pi * fx)) * sobel(uwpx, axis=1)
    slope_y = (1 / (2 * np.pi * fy)) * sobel(uwpy, axis=0)
  

    # k = np.array([[-1,0,1]])
    # slope_x = (1 / (2 * np.pi * fx)) * convolve(uwpx, k)
    # slope_y = (1 / (2 * np.pi * fy)) * convolve(uwpy, np.transpose(k))

    height_map = np.cumsum(slope_x, axis=1) + np.cumsum(slope_y, axis=0)

    return height_map

image_path = 'U-Net/2D_fringe/2d_obj_Sphere_view1.png' #'U-Net/parabolic_concave/2d_obj_Plane.002_view0.png'

image = Image.open(image_path).convert('L')
height_map = test_phase_difference(image)

height_map = (height_map - np.min(height_map)) / np.ptp(height_map)

def plot_depth_map_3d(depth_map):
        X, Y = np.meshgrid(np.arange(depth_map.shape[1]), np.arange(depth_map.shape[0]))

        # Plot the depth map in 3D
        fig = plt.figure()
        ax = fig.add_subplot(111, projection='3d')
        ax.plot_surface(X, Y, depth_map, cmap='viridis')  
        ax.set_xlabel('Width')
        ax.set_ylabel('Height')
        ax.set_zlabel('Depth')
        ax.set_title('Depth Map Visualization')
        plt.show()


# plot_depth_map_3d(height_map)


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
def depth_map_normalizer(depthMap):

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

    return normalized_depthmap



import os
from sklearn.metrics import mean_squared_error, mean_absolute_error

def calculate_metrics(predicted, ground_truth):
    predicted = predicted.flatten()
    ground_truth = ground_truth.flatten()

    mse = mean_squared_error(ground_truth, predicted)
    mae = mean_absolute_error(ground_truth, predicted)
    rmse = np.sqrt(mse)
    log_error = np.sqrt(mean_squared_error(np.log1p(ground_truth+1), np.log1p(predicted+1)))
    return mse, mae, rmse, log_error

folder_path = 'U-Net/parabolic_concave'
image_extension = '.png'
depth_extension = '.npz'

mse_list = []
mae_list = []
rmse_list = []
log_error_list = []
from tqdm import tqdm

for filename in tqdm(os.listdir(folder_path), desc="Processing files"):

    if filename.endswith(image_extension):
        image_path = os.path.join(folder_path, filename)
        depth_filename = filename.replace('2d_obj_', 'depth_obj_').replace('.png', '.npz')
        depth_path = os.path.join(folder_path, depth_filename)
        image = Image.open(image_path).convert('L')
        height_map = test_phase_difference(image)
        height_map = (height_map - np.min(height_map)) / np.ptp(height_map)

        ground_truth = np.load(depth_path)['arr_0']
        ground_truth = depth_map_normalizer(ground_truth)
 

        height_map_flat = height_map.flatten()
        ground_truth_flat = ground_truth.flatten()

        # plot_depth_map_3d(height_map)
        # plot_depth_map_3d(ground_truth)

        mse, mae, rmse, log_error = calculate_metrics(height_map_flat, ground_truth_flat)
        
        mse_list.append(mse)
        mae_list.append(mae)
        rmse_list.append(rmse)
        log_error_list.append(log_error)

average_mse = np.mean(mse_list)
average_mae = np.mean(mae_list)
average_rmse = np.mean(rmse_list)
average_log_error = np.mean(log_error_list)

print(f"Average MSE: {average_mse}")
print(f"Average MAE: {average_mae}")
print(f"Average RMSE: {average_rmse}")
print(f"Average Log Error: {average_log_error}")

