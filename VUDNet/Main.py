from Surface_VUD import Surface_VH
import argparse
import torch
# from UNet import UNet
import torch.nn as nn
import torch.optim as optim 
from tqdm import tqdm
from torchvision import transforms
from torch.utils.data import random_split
from keras.models import load_model
from keras import backend as Kend
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from torch.utils.tensorboard import SummaryWriter



device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print ("Running on "+ str(device))
print("GPU available:", tf.config.list_physical_devices('GPU'))
torch.cuda.empty_cache()


transform = transforms.Compose([
    transforms.ToTensor(),
    # transforms.Resize((, )),
    transforms.Normalize(mean=[0.0], std=[0.5])
])

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

def plot_images_from_loader(data_loader):
    for wpx, wpy, depth_map_resized_np in data_loader:
        # Plot the images
        for i in range(len(wpx)):
            fig, axs = plt.subplots(1, 2, figsize=(10, 5))

            # Plot wpx
            axs[0].imshow(wpx[i].squeeze(), cmap='gray')
            axs[0].set_title('wpx')
            axs[0].axis('off')

            # Plot wpy
            axs[1].imshow(wpy[i].squeeze(), cmap='gray')
            axs[1].set_title('wpy')
            axs[1].axis('off')

            plt.show()

            # Plot depth map
            plot_depth_map_3d(depth_map_resized_np[i])

def Main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--dataset_path",
        type=str ,
        default= "parabolic_concave", #'2D_fringe_test',#"parabolic_concave",
        help="Dir to the dataset consisting of images and depthMap (in .npz format)"
    )
    parser.add_argument(
        "--image_suffix",
        type=str,
        default=".png"
    )
    parser.add_argument(
        "--depthMap_suffix",
        type=str,
        default=".npz"
    )
    parser.add_argument(
        "--lr",
        type=float,
        default=0.001
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=5
    )
    parser.add_argument(
        "--num_epochs",
        type=int,
        default=100
    )
    parser.add_argument(
        "--height",
        type=int,
        default=256
    )
    parser.add_argument(
        "--width",
        type=int,
        default=256
    )
    parser.add_argument(
        "--TrainSplit",
        type=float,
        default=0.8
    )
    parser.add_argument(
        "--ValidationSplit",
        type=float,
        default=0.1
    )
    parser.add_argument(
        "--Augmentation",
        type=bool,
        default=True
    )
    options = parser.parse_args()

    datset_path = options.dataset_path
    image_suffix = options.image_suffix
    depthMap_suffix = options.depthMap_suffix
    lr = options.lr
    batch_size = options.batch_size
    num_epochs = options.num_epochs
    height = options.height
    width = options.width
    TrainSplit = options.TrainSplit
    ValidationSplit = options.ValidationSplit
    Augmentation_flag = options.Augmentation # TO-DO
    ###################################################################################


    Kend.clear_session()
    DYModel = load_model('./DYnet++.h5', custom_objects={"tf": tf})


    dataset = Surface_VH(datset_path, image_suffix, depthMap_suffix, transform, DYModel)
    data_loader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=True)

    # plot_images_from_loader(data_loader)

    train_size = int(TrainSplit * len(dataset))   
    val_size = int(ValidationSplit * len(dataset))    
    test_size = len(dataset) - train_size - val_size   

    train_set, val_set, test_set = random_split(dataset, [train_size, val_size, test_size])

    train_loader = torch.utils.data.DataLoader(train_set, batch_size=batch_size, shuffle=True)
    val_loader = torch.utils.data.DataLoader(val_set, batch_size=batch_size, shuffle=False)
    test_loader = torch.utils.data.DataLoader(test_set, batch_size=batch_size, shuffle=False)



    options ={
    # -- U-Net Options -- #
    'unet_conv_filters': [16,32,64,128,256],  # Number of filters in the U-Net.
    'conv_kernel_size': (3, 3),  # Size of convolutional kernels.
    'conv_stride_rate': (1, 1),  # Stride rate of convolutional kernels.
    'conv_dilation_rate': (1, 1),  # Dilation rate of convolutional kernels.
    'conv_padding': (1, 1),  # Number of padded pixels in convolutional layers.
    'conv_padding_style': 'zeros',  # Style of padding.
    'n_classes': 1 ,# Number of output channels
    'input_channels': 9 # we have two gray scale thingy, wpx and wpy (instead of RGB being channel 3) # now 9 :D
    }

    model =   UNet(options).to(device) # NUNet().to(device) # UNet().to(device)

    criterion = nn.MSELoss()
    # criterion = torch.nn.SmoothL1Loss()

    # def depth_loss(y_true, y_pred):
    #     return torch.mean(torch.abs(torch.log(y_true) - torch.log(y_pred)))
    # criterion = depth_loss

    # def berhu_loss(y_true, y_pred, c=0.2):
    #     abs_diff = torch.abs(y_true - y_pred)
    #     max_diff = torch.max(abs_diff)
    #     if max_diff <= c:
    #         return torch.mean(abs_diff)
    #     else:
    #         return torch.mean((abs_diff ** 2 + c ** 2) / (2 * c))

    # criterion = berhu_loss

    # def scale_invariant_loss(y_true, y_pred):
    #     return torch.mean(torch.abs((y_true - y_pred) / y_true))

    optimizer = optim.Adam(model.parameters(), lr=lr)
    # optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate, momentum=0.9)
    # optimizer = torch.optim.RMSprop(model.parameters(), lr=learning_rate)
    # optimizer = torch.optim.Adadelta(model.parameters(), lr=learning_rate)
    # optimizer = torch.optim.Adagrad(model.parameters(), lr=learning_rate)
    # optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)


    # scheduler_step_lr = lr_scheduler.StepLR(optimizer, step_size=gamma, gamma=0.1)
    # scheduler_multi_step_lr = lr_scheduler.MultiStepLR(optimizer, milestones=[epoch1, epoch2], gamma=0.1)
    # scheduler_exp_lr = lr_scheduler.ExponentialLR(optimizer, gamma=0.95)
    # scheduler_cosine_annealing_lr = lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)
    # scheduler_reduce_lr_on_plateau = lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.1, patience=10, verbose=True)
    # scheduler_cyclic_lr = lr_scheduler.CyclicLR(optimizer, base_lr=0.001, max_lr=0.1, step_size_up=2000)
    # scheduler_one_cycle_lr = lr_scheduler.OneCycleLR(optimizer, max_lr=0.1, steps_per_epoch=len(train_loader), epochs=num_epochs)


    writer = SummaryWriter('runs/experiment_1')


    model.train()
    for epoch in range(num_epochs):
        running_loss = 0.0
        with tqdm(train_loader, desc=f'Epoch {epoch+1}/{num_epochs}', unit='batch') as t:
            for i, (images, depth_maps) in enumerate(t):
                images, depth_maps = images.to(device), depth_maps.to(device).float()
                optimizer.zero_grad()
                outputs = model(images) 
                outputs = outputs.squeeze(1)
                loss = criterion(outputs, depth_maps)
                # loss = custom_loss_function(outputs, depth_maps, height, width)
                loss.backward()
                optimizer.step()
                running_loss += loss.item()
                writer.add_scalar('training loss', running_loss / (i+1), epoch * len(train_loader) + i)
        print(f'Epoch {epoch + 1}, Loss: {running_loss / len(data_loader)}')               
    print('Finished Training')

    writer.close()
    # # Training phase
    # model.train()
    # for epoch in range(num_epochs):
    #     running_loss = 0.0
    #     with tqdm(train_loader, desc=f'Epoch {epoch+1}/{num_epochs}', unit='batch') as t:
    #         for i, (wpx_wpy, depth_maps) in enumerate(t):
    #             wpx_wpy, depth_maps = wpx_wpy.to(device), depth_maps.to(device)
    #             wpx, wpy = torch.split(wpx_wpy, 1, dim=1)  # Split the stacked input into wpx and wpy
    #             optimizer.zero_grad()
    #             outputs = model(torch.cat((wpx, wpy), dim=1))  # Concatenate wpx and wpy along the channel dimension
    #             loss = criterion(outputs, depth_maps)
    #             loss.backward()
    #             optimizer.step()
    #             running_loss += loss.item()
    #     print(f'Epoch {epoch + 1}, Loss: {running_loss / len(train_loader)}')               
    # print('Finished Training')



    # Validation phase
    model.eval()
    val_loss = 0.0
    with torch.no_grad():
        for images_val, depth_maps_val in val_loader:
            images_val, depth_maps_val = images_val.to(device), depth_maps_val.to(device).float()
            outputs_val = model(images_val)
            loss_val = criterion(outputs_val, depth_maps_val)
            val_loss += loss_val.item()
    print(f'Validation - Epoch {epoch + 1}, Loss: {val_loss / len(val_loader)}')
    
    # Testing phase
    model.eval()
    test_loss = 0.0
    with torch.no_grad():
        for images_test, depth_maps_test in test_loader:
            images_test, depth_maps_test = images_test.to(device), depth_maps_test.to(device).float()
            outputs_test = model(images_test)
            loss_test = criterion(outputs_test, depth_maps_test)
            test_loss += loss_test.item()
    print(f'Test Loss: {test_loss / len(test_loader)}')


    import matplotlib.pyplot as plt

    # Set model to evaluation mode
    model.eval()

    # Iterate through the test set and get predictions
    with torch.no_grad():
        for images_test, depth_maps_test in test_loader:
            images_test, depth_maps_test = images_test.to(device), depth_maps_test.to(device)
            outputs_test = model(images_test)

            # Convert tensors to numpy arrays
            images_test_np = images_test.cpu().numpy()
            depth_maps_test_np = depth_maps_test.cpu().numpy()
            outputs_test_np = outputs_test.cpu().numpy()

            for i in range(len(images_test_np)):
                plt.figure(figsize=(45, 5))

                # Plot first channel
                plt.subplot(1, 12, 1)
                plt.imshow(images_test_np[i][0], cmap='gray')
                plt.title('wpx')
                plt.axis('off')

                # Plot second channel
                plt.subplot(1, 12, 2)
                plt.imshow(images_test_np[i][1], cmap='gray')
                plt.title('wpy')
                plt.axis('off')

                plt.subplot(1, 12, 3)
                plt.imshow(images_test_np[i][2], cmap='gray')
                plt.title('image')
                plt.axis('off')
                plt.subplot(1, 12, 4)
                plt.imshow(images_test_np[i][3], cmap='gray')
                plt.title('magnitude_img')
                plt.axis('off')

                plt.subplot(1, 12, 5)
                plt.imshow(images_test_np[i][4], cmap='gray')
                plt.title('direction_img')
                plt.axis('off')

                plt.subplot(1, 12, 6)
                plt.imshow(images_test_np[i][5], cmap='gray')
                plt.title('magnitude_wpx')
                plt.axis('off')

                plt.subplot(1, 12, 7)
                plt.imshow(images_test_np[i][6], cmap='gray')
                plt.title('direction_wpx')
                plt.axis('off')

                plt.subplot(1, 12, 8)
                plt.imshow(images_test_np[i][7], cmap='gray')
                plt.title('magnitude_wpy')
                plt.axis('off')

                plt.subplot(1, 12, 9)
                plt.imshow(images_test_np[i][8], cmap='gray')
                plt.title('direction_wpy')
                plt.axis('off')


                plt.subplot(1, 12, 10)
                plt.imshow(images_test_np[i][8], cmap='gray')
                plt.title('Mere repeat')
                plt.axis('off')

                # Plot ground truth depth map
                plt.subplot(1, 12, 11)
                plt.imshow(depth_maps_test_np[i].squeeze(), cmap='gray')
                plt.title('Ground Truth')
                plt.axis('off')

                # Plot predicted depth map
                plt.subplot(1, 12, 12)
                plt.imshow(outputs_test_np[i].squeeze(), cmap='gray')
                plt.title('Predicted')
                plt.axis('off')

                plt.show()
                
                plot_depth_map_3d(depth_maps_test_np[i].squeeze())
                plot_depth_map_3d(outputs_test_np[i].squeeze())
                


Main()
 
