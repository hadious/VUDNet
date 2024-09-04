from Surface_VH import Surface_Dataset
import argparse
import torch
from UNet import UNet
import torch.nn as nn
import torch.optim as optim 
from tqdm import tqdm
from torchvision import transforms
from torch.utils.data import random_split

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print ("Running on "+ str(device))
torch.cuda.empty_cache()

transform = transforms.Compose([
    transforms.ToTensor(),
    # transforms.Resize((, )),
    transforms.Normalize(mean=[0.0, 0.0, 0.0], std=[0.5, 0.5, 0.5])
])

def custom_loss_function(output, target, height, width):
    di = target - output
    n = (height *  width)
    di2 = torch.pow(di, 2)
    fisrt_term = torch.sum(di2,(1,2,3))/n
    second_term = 0.5*torch.pow(torch.sum(di,(1,2,3)), 2)/ (n**2)
    loss = fisrt_term - second_term
    return loss.mean()

def Main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--dataset_path",
        type=str ,
        default="Convex_1Sphere_plane",
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
        default=3
    )
    parser.add_argument(
        "--num_epochs",
        type=int,
        default=5
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
        default=0.15
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

    dataset = Surface_Dataset(datset_path, image_suffix, depthMap_suffix, transform) 
    data_loader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=True)
 

    train_size = int(TrainSplit * len(dataset))   
    val_size = int(ValidationSplit * len(dataset))    
    test_size = len(dataset) - train_size - val_size   

    train_set, val_set, test_set = random_split(dataset, [train_size, val_size, test_size])

    train_loader = torch.utils.data.DataLoader(train_set, batch_size=batch_size, shuffle=True)
    val_loader = torch.utils.data.DataLoader(val_set, batch_size=batch_size, shuffle=False)
    test_loader = torch.utils.data.DataLoader(test_set, batch_size=batch_size, shuffle=False)



    options ={
    # -- U-Net Options -- #
    'unet_conv_filters': [64,128,256,512,1024],  # Number of filters in the U-Net.
    'conv_kernel_size': (7, 7),  # Size of convolutional kernels.
    'conv_stride_rate': (1, 1),  # Stride rate of convolutional kernels.
    'conv_dilation_rate': (1, 1),  # Dilation rate of convolutional kernels.
    'conv_padding': (1, 1),  # Number of padded pixels in convolutional layers.
    'conv_padding_style': 'zeros',  # Style of padding.
    'n_classes': 1, # Number of output channels
    'input_channels': 3
    }

    model =   UNet(options).to(device) # NUNet().to(device) # UNet().to(device)

    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=lr)

    model.train()
    for epoch in range(num_epochs):
        running_loss = 0.0
        with tqdm(train_loader, desc=f'Epoch {epoch+1}/{num_epochs}', unit='batch') as t:
            for i, (images, depth_maps) in enumerate(t):
                images, depth_maps = images.to(device), depth_maps.to(device)
                optimizer.zero_grad()
                outputs = model(images)
                loss = criterion(outputs, depth_maps)
                # loss = custom_loss_function(outputs, depth_maps, height, width)
                loss.backward()
                optimizer.step()
                running_loss += loss.item()
        print(f'Epoch {epoch + 1}, Loss: {running_loss / len(data_loader)}')               
    print('Finished Training')


    # Validation phase
    model.eval()
    val_loss = 0.0
    with torch.no_grad():
        for images_val, depth_maps_val in val_loader:
            images_val, depth_maps_val = images_val.to(device), depth_maps_val.to(device)
            outputs_val = model(images_val)
            loss_val = criterion(outputs_val, depth_maps_val)
            val_loss += loss_val.item()
    print(f'Validation - Epoch {epoch + 1}, Loss: {val_loss / len(val_loader)}')
    
    # Testing phase
    model.eval()
    test_loss = 0.0
    with torch.no_grad():
        for images_test, depth_maps_test in test_loader:
            images_test, depth_maps_test = images_test.to(device), depth_maps_test.to(device)
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

            # Plot the images, ground truth, and predictions
            for i in range(len(images_test_np)):
                plt.figure(figsize=(10, 5))

                plt.subplot(1, 3, 1)
                plt.imshow(images_test_np[i].transpose(1, 2, 0))
                plt.title('Input Image')
                plt.axis('off')

                plt.subplot(1, 3, 2)
                plt.imshow(depth_maps_test_np[i].squeeze(), cmap='gray')
                plt.title('Ground Truth Depth Map')
                plt.axis('off')

                plt.subplot(1, 3, 3)
                plt.imshow(outputs_test_np[i].squeeze(), cmap='gray')
                plt.title('Predicted Depth Map')
                plt.axis('off')

                plt.show()
    

Main()

