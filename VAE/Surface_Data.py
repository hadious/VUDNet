from torch.utils.data import Dataset
import os 
from PIL import Image
import numpy as np 
from scipy.ndimage import convolve
from sklearn.decomposition import PCA
from PIL import ImageOps


GAN = False
INPUT_CHANNELS = "Nine"

class Surface_Data(Dataset):

    def __init__(self, path,  image_suffix, depthMap_suffix, transform, DYModel):
        self.path = path 
        self.image_suffix = image_suffix 
        self.depthMap_suffix = depthMap_suffix
        self.image_files = [f for f in os.listdir(path) if f.endswith(image_suffix)]
        self.transform = transform
        self.DYModel = DYModel
        
    
    def __len__(self):
         return len(self.image_files)
    
    def gradient(self,image):
        # Convert the image to grayscale if it's RGB
        if len(image.shape) == 3:
            image = np.mean(image, axis=2)
        
        # Sobel filters for gradient calculation
        sobel_x = np.array([[-1, 0, 1],
                            [-2, 0, 2],
                            [-1, 0, 1]])
        sobel_y = np.array([[-1, -2, -1],
                            [0, 0, 0],
                            [1, 2, 1]])
        
        gradient_x = convolve(image, sobel_x)
        gradient_y = convolve(image, sobel_y)
        
        gradient_magnitude = np.sqrt(gradient_x**2 + gradient_y**2)
        
        gradient_direction = np.arctan2(gradient_y, gradient_x)
        
        return gradient_magnitude, gradient_direction
    
    @staticmethod
    def calculate_centroid(matrix):
        return np.mean(matrix, axis=0)

    def costume_test_phase_difference(self, img):

        img_height = 320
        img_width = 240
        test_x = np.empty((1, img_height, img_width, 1), dtype="float32")
        img = img.crop((500, 50, 1400, 900))
        img_resized = img.resize((img_width, img_height))
        img_array = np.array(img_resized, dtype=np.float32)
        img_normalized = img_array / 255.0
        test_x[0, :, :, 0] = img_normalized
        

        (prediction_M, prediction_D) = self.DYModel.predict(test_x)
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

        return wpx, wpy, img_normalized

    @staticmethod
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
    
    @staticmethod
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

    def __getitem__(self, index):
        image_name = self.image_files[index]
        image_path = os.path.join(self.path, image_name)
        depth_map_path = os.path.join(self.path, image_name.replace(self.image_suffix, self.depthMap_suffix))
        depth_map_path =  depth_map_path.replace("2d", "depth")
        
        # Load image
        image = Image.open(image_path).convert('L')
        
        # image = self.transform(image)
        wpx, wpy, img = self.costume_test_phase_difference(image)
        img = np.array(img)

        magnitude_img, direction_img = self.gradient(img)
        magnitude_wpx, direction_wpx = self.gradient(wpx)
        magnitude_wpy, direction_wpy = self.gradient(wpy)

        desired_size = (512, 256)  
        zero_padded_images = []
        padding = ((desired_size[0] - 320) // 2, (desired_size[1] - 240) // 2)

        if GAN == True:
                
            for image in (wpx, wpy, img, magnitude_img, direction_img, magnitude_wpx, direction_wpx, magnitude_wpy, direction_wpy):
                
                image = np.array(image)
                # print (image.shape)
                # padded_image = ImageOps.expand(image, padding, fill=0)    
                # print (padding)
                padded_image = np.pad(image,((padding[0],padding[0]),(padding[1],padding[1])), mode='constant', constant_values=0)
                # print (padded_image.shape)
                zero_padded_images.append(padded_image)
            wpx_wpy_img = np.stack(zero_padded_images, axis=0)
        else:
            if INPUT_CHANNELS == "Nine":
                wpx_wpy_img = np.stack((wpx, wpy, img, magnitude_img, direction_img, magnitude_wpx, direction_wpx, magnitude_wpy, direction_wpy), axis=0)
            elif INPUT_CHANNELS == "One":
                wpx_wpy_img = img
            elif INPUT_CHANNELS == "Three":
                wpx_wpy_img = np.stack((wpx, wpy, img), axis=0)


        # Load depth map
        depth_data = np.load(depth_map_path)
        depthMap = depth_data['arr_0']
        # normalized_depthmap = (depthMap - depthMap.mean()) / depthMap.std()

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

        centroid = self.calculate_centroid(depth_matrix)

        pca = PCA(n_components=3)
        pca.fit(depth_matrix)
        normal_vector = pca.components_[2]  

        yaw, pitch, roll = self.calculate_rotation_angles(normal_vector)
        yaw = 0
        pitch = 0
        roll = roll if roll > 0 else roll + 180 
        rotated_matrix = self.rotate_matrix_3d(depth_matrix, roll, pitch, yaw, centroid)
        rotated_matrix = self.rotate_matrix_3d(rotated_matrix, 0, 0, 180, centroid)

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
        # import pdb
        # pdb.set_trace()
        if GAN == True:
            padded_array_2d = np.pad(array_2d, ((padding[0],padding[0]),(padding[1],padding[1])) , mode='constant', constant_values=0)
            transposed_array = wpx_wpy_img.transpose(1, 2, 0)
            padded_array_2d = np.expand_dims(padded_array_2d, axis=-1)        
            transposed_array = np.expand_dims(transposed_array[:,:,2], axis=-1) # JUSTTTT for 1 input, remove for 9 channels plzzzzzzzz
            return transposed_array, padded_array_2d
        return wpx_wpy_img, normalized_depthmap
