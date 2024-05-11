'''
import numpy as np
import matplotlib.pyplot as plt

# Load the .npz file
data = np.load('2D_fringe/depth_obj_Sphere_view1.npz')

# Access the depth map array
image_data = data['arr_0']  # Replace 'arr_0' with the key name if it's different
print(image_data.shape)
print (np.mean(image_data))
# image_data = image_data.reshape(1080, 1920, 4)
# Visualize the depth map

plt.imshow(image_data)  # You can change the colormap as needed
plt.colorbar(label='Depth')  # Add a colorbar to indicate depth values
plt.title('Depth Map Visualization')
plt.xlabel('Width')
plt.ylabel('Height')
plt.show()  
'''





# from PIL import Image

# def create_stripe_image(width, height, stripe_width, color1, color2):

#     image = Image.new("RGB", (width, height))
#     pixels = image.load()

#     for y in range(height):
#         for x in range(width):
#             if x % (stripe_width * 2) < stripe_width:
#                 pixels[x, y] = color1
#             else:
#                 pixels[x, y] = color2

#     return image

# if __name__ == "__main__":
#     # Define image dimensions and stripe properties
#     width = 800
#     height = 600
#     stripe_width = 10
#     color1 = (255, 255, 255)  # White
#     color2 = (0, 0, 0)         # Black

#     # Create stripe image
#     stripe_image = create_stripe_image(width, height, stripe_width, color1, color2)

#     # Save image as "pattern.jpg"
#     stripe_image.save("/home/hadi/Desktop/pattern.jpg")




# import numpy as np
# import matplotlib.pyplot as plt
# from PIL import Image

# # Load the .npz file
# depth_data = np.load('2D_fringe/depth_obj_Sphere_view1.npz')
# depthMap = depth_data['arr_0']
# # normalized_depthmap = (depthMap - depthMap.mean()) / depthMap.std()

# depth_map_img = Image.fromarray(depthMap)
# depth_map_cropped_img = depth_map_img.crop((500, 50, 1400, 900))
# depth_map_resized_img = depth_map_cropped_img.resize((320, 240))
# depth_map_resized_np = np.array(depth_map_resized_img)

# print (np.min(depth_map_resized_np[150]))

# plt.imshow(depth_map_resized_np,cmap='viridis')  # You can change the colormap as needed
# plt.colorbar(label='Depth')  # Add a colorbar to indicate depth values
# plt.title('Depth Map Visualization')
# plt.xlabel('Width')
# plt.ylabel('Height')
# plt.show()  


import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D  # Import Axes3D from mpl_toolkits.mplot3d
from PIL import Image

# Load the .npz file
depth_data = np.load('2D_fringe/depth_obj_Sphere_view1.npz')
depthMap = depth_data['arr_0']

# Crop and resize the depth map
depth_map_img = Image.fromarray(depthMap)
depth_map_cropped_img = depth_map_img.crop((500, 50, 1400, 900))
depth_map_resized_img = depth_map_cropped_img.resize((320, 240))
depth_map_resized_np = np.array(depth_map_resized_img)

# Create meshgrid for 3D plotting
X, Y = np.meshgrid(np.arange(depth_map_resized_np.shape[1]), np.arange(depth_map_resized_np.shape[0]))

# Plot the depth map in 3D
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
ax.plot_surface(X, Y, depth_map_resized_np, cmap='viridis')  # Use 'viridis' colormap for better perception of depth
ax.set_xlabel('Width')
ax.set_ylabel('Height')
ax.set_zlabel('Depth')
ax.set_title('Depth Map Visualization')
plt.show()