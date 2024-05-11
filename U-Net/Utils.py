# from PIL import Image
# from scipy.io import savemat
# import numpy as np


# image = Image.open("2d_obj_sphere_view1.png")
# resized_image = image.resize((240, 320))
# image_array = np.array(resized_image)
# data = {'image': image_array}
# savemat('output_image.mat', data)


import scipy.io
import matplotlib.pyplot as plt

# Load the .mat file
mat_data = scipy.io.loadmat('IIIC/Ic_1.mat')

# Access the image array from the loaded data
image_array = mat_data['image']

# Plot the image
plt.imshow(image_array, cmap='gray')  # Assuming it's a grayscale image
plt.axis('off')  # Turn off axis
plt.show()
