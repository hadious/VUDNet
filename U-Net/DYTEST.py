# -*- coding: utf-8 -*-
"""
This Python script is used for verifying DYnet++ model with the following test data:
    IIIA: Test data in the section III.A
    IIIB: Test data in the section III.B: Low reflectivity and defects patterns
    IIIC: Test data in the section III.C: Complex patterns with closed- and opened- loop
The input of the test data is 'Ic' means composite pattern
The ground truth is 16-step phase-shifting method phase with x-phase = arctan2(Mx,My), y-phase = arctan2(Dx,Dy)
We expect that by inputing the single composite pattern Ic, our DYnet++ could generate similar output like ground-truth.
    
$How to test:
    1. Change the name of 'dir_base' to 'IIIA, IIIB or IIIC'
    2. Run the script
    3. Run the function 'test_phase_difference(frame)' with frame is the pattern number (1 or 2)
$How to check the results:
    The output results are figure of input single composite pattern, retrived wrapped phase of DYnet++ and phase shifting, unwrapped phase comparison
    
"""
from keras import backend as Kend
import tensorflow as tf
import numpy as np
import h5py
import matplotlib.pyplot as plt
from keras.models import load_model
from skimage.restoration import unwrap_phase
from PIL import Image

# Test image size
img_height = 320
img_width = 240


Kend.clear_session()
model = load_model('./DYnet++.h5', custom_objects={"tf": tf})


dir_base = './IIIA/'
test_dir = dir_base +'Ic_'

# Calculate the root-mean-square error
def rms(X):
    square = np.square(X)
    mse = square.mean()
    rmse = np.sqrt(mse)
    return rmse


def costume_test_phase_difference(frame):


    test_dir = "2d_obj_Plane.002_view0.png"

    img_height = 320
    img_width = 240

    test_x = np.empty((1, img_height, img_width, 1), dtype="float32")

    img = Image.open(test_dir).convert('L')  # Open the image and convert to grayscale


    img = img.crop((500, 50, 1400, 900))
    # print(img.size)
    


    img_resized = img.resize((img_width, img_height))

    img_array = np.array(img_resized, dtype=np.float32)

    img_normalized = img_array / 255.0

    test_x[0, :, :, 0] = img_normalized

    plt.figure()
    plt.imshow(img_normalized, cmap='gray')
    plt.show()
    


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


    plt.figure()
    plt.imshow(wpx, cmap='gray')
    plt.show()


    plt.figure()
    plt.imshow(wpy, cmap='gray')
    plt.show()



# test_phase_difference(1)
# print ('here')

costume_test_phase_difference(1)

# from keras.utils.vis_utils import plot_model

# model.summary()
# plot_model(model, to_file='model_plot.png', show_shapes=True, show_layer_names=True)










# Display the phase difference
def test_phase_difference(frame):
    test_x = np.empty((1, img_height ,img_width,1),dtype="float32")
    
    img_add = test_dir + str(frame) +'.mat'    
    mat_train = h5py.File(img_add)  
    img = mat_train['Ic']
    test_x[0, :, :, 0] = img[:]/255
    plt.figure(1)
    plt.imshow(img,cmap='gray')
    
    # Get groundtruth wrapped phase
    M_mat = h5py.File(dir_base+'Mx_'+str(frame)+'.mat')
    M_data = M_mat['Mx']
    D_mat = h5py.File(dir_base+'Dx_'+str(frame)+'.mat')
    D_data = D_mat['Dx']
    M_data = M_data[:]
    D_data = D_data[:]
    wpx_ground_truth = np.arctan2(M_data,D_data)
    
    M_mat = h5py.File(dir_base+'My_'+str(frame)+'.mat')
    M_data = M_mat['My']
    D_mat = h5py.File(dir_base+'Dy_'+str(frame)+'.mat')
    D_data = D_mat['Dy']
    M_data = M_data[:]
    D_data = D_data[:]
    wpy_ground_truth = np.arctan2(M_data,D_data)
  
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
    
    # Unwrap phase and compare
    uwpx = unwrap_phase(wpx)
    uwpy = unwrap_phase(wpy)
    uwpx_ground = unwrap_phase(wpx_ground_truth)
    uwpy_ground = unwrap_phase(wpy_ground_truth)
    phase_diff_x = uwpx-uwpx_ground
    phase_diff_y = uwpy-uwpy_ground
    
    phase_diff_x = phase_diff_x - np.mean(phase_diff_x)
    phase_diff_y = phase_diff_y - np.mean(phase_diff_y)
    
    # Display phase difference PV value
    PVx = np.max(phase_diff_x)-np.min(phase_diff_x)
    PVy = np.max(phase_diff_y)-np.min(phase_diff_y)
    PVx = round(PVx,2)
    PVy = round(PVy,2)
    
    rmse_x = rms(phase_diff_x)
    rmse_x = round(rmse_x,2)
    
    rmse_y = rms(phase_diff_y)
    rmse_y = round(rmse_y,2)
    
    # Display wrapped phase
    plt.figure(2)
    fig,ax = plt.subplots(2,2)
    ax[0,0].imshow(wpx_ground_truth,cmap='gray')
    ax[0,0].set_title('PS x wrapped phase')
    ax[0,1].imshow(wpy_ground_truth,cmap='gray')
    ax[0,1].set_title('PS y wrapped phase')
    ax[1,0].imshow(wpx,cmap='gray')
    ax[1,0].set_title('DL x wrapped phase')
    ax[1,1].imshow(wpy,cmap='gray')
    ax[1,1].set_title('DL y wrapped phase')
    fig.tight_layout()
    plt.show()
    # Display unwrapped phase difference
    plt.figure(3)
    fig,ax = plt.subplots(2)
    pdx = ax[0].imshow(phase_diff_x,cmap='jet')
    ax[0].set_title('PV = '+str(PVx)+' rad' +'\n'+ 'RMS = '+str(rmse_x)+' rad')
    plt.colorbar(pdx,ax = ax[0])
    pdy = ax[1].imshow(phase_diff_y,cmap='jet')
    ax[1].set_title('PV = '+str(PVy)+' rad' +'\n'+ 'RMS = '+str(rmse_y)+' rad')
    plt.colorbar(pdy,ax = ax[1])
    fig.tight_layout()
    plt.show()