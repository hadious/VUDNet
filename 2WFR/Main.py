import numpy as np
import cv2
from scipy.ndimage import gaussian_filter
from scipy.integrate import simps
from skimage.filters import window

def windowed_fourier_ridges(image, window_size):
    """
    Perform windowed Fourier transform and extract ridges on-the-fly.
    """
    rows, cols = image.shape
    half_window = window_size // 2
    ridges_x = np.zeros((rows, cols))
    ridges_y = np.zeros((rows, cols))

    hann_window = window('hann', (window_size, window_size))

    for i in range(half_window, rows - half_window):
        for j in range(half_window, cols - half_window):
            windowed_image = image[i - half_window:i + half_window, j - half_window:j + half_window].astype(float)
            if windowed_image.shape == hann_window.shape:
                windowed_image *= hann_window
                ft_windowed_image = np.fft.fftshift(np.fft.fft2(windowed_image))

                magnitude = np.abs(ft_windowed_image)
                ridges_x[i, j] = np.argmax(magnitude, axis=0).mean()
                ridges_y[i, j] = np.argmax(magnitude, axis=1).mean()

    return ridges_x, ridges_y

def integrate_ridges(ridges_x, ridges_y):
    """
    Integrate the ridges over x and y to reconstruct the 3D surface.
    """
    height, width = ridges_x.shape
    integral_x = np.zeros((height, width))
    integral_y = np.zeros((height, width))

    for i in range(height):
        integral_x[i] = simps(ridges_x[i], dx=1)

    for j in range(width):
        integral_y[:, j] = simps(ridges_y[:, j], dx=1)

    reconstruction = (integral_x + integral_y) / 2
    return reconstruction

def reconstruct_surface(image, window_size=32):
    """
    Perform the 2WFR process and reconstruct the 3D surface.
    """
    ridges_x, ridges_y = windowed_fourier_ridges(image, window_size)
    reconstruction = integrate_ridges(ridges_x, ridges_y)
    return reconstruction

if __name__ == "__main__":
    # Load the image (example image)
    image = cv2.imread('U-Net/Convex_1Sphere_plane/2d_obj_Sphere_view1.png', cv2.IMREAD_GRAYSCALE)
    image = cv2.resize(image, (500,500)) 
    image = gaussian_filter(image, sigma=1)  # Preprocess with Gaussian filter

    # Reconstruct the 3D surface
    reconstructed_surface = reconstruct_surface(image)

    # Save or display the reconstructed surface
    np.save('reconstructed_surface.npy', reconstructed_surface)
    cv2.imshow('Reconstructed Surface', reconstructed_surface)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
