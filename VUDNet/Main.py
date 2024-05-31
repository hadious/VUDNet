import yaml
from torchvision import transforms
import VUDNet
import torch
import tensorflow as tf
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim 
from tqdm import tqdm
from Utils import  Load_the_dataset, plot_depth_map_3d, calculate_metrics, visualize_latent_space
import numpy as np
import os


device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print ("Running on "+ str(device))
print("GPU available:", tf.config.list_physical_devices('GPU'))
torch.cuda.empty_cache()

# Configs:
##########################################
with open('VUDNet/Config.yaml', 'r') as file:
    config = yaml.safe_load(file)

Run_VAE = config['RunTime']['Run_VAE']
Run_UNet = config['RunTime']['Run_UNet']
VAE_configs = config['VAE']
Dataset_configs = config['DataLoader']
VAE_lr = config['VAE']['lr']
UNET_lr = config['UNet']['lr']
num_epochs = config['RunTime']['epoch_number']
fusion_epochs = config['RunTime']['fusion_epochs']
fusion_lr = config['RunTime']['lr']
Show_PLOT_RESULTS = config['RunTime']['Show_PLOT_RESULTS']
CALCULATE_THE_CRITERIA = config['RunTime']['CALCULATE_THE_CRITERIA']
SAVE_DEPTH_MAP_RESULTS = config['RunTime']['SAVE_DEPTH_MAP_RESULTS']
NPY_SAVE_PATH = config['RunTime']['NPY_SAVE_PATH']
VISUALIZE_THE_DIST = config['RunTime']['VISUALIZE_THE_DIST']
##########################################

# Data Loading:
##########################################

transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.0], std=[0.5])
])

train_loader, val_loader, test_loader = Load_the_dataset(Dataset_configs, transform)

##########################################

# Model Loading:
##########################################

Unet_option ={
    'unet_conv_filters': [16,32,64,128,256],  # Number of filters in the U-Net.
    'conv_kernel_size': (3, 3),  # Size of convolutional kernels.
    'conv_stride_rate': (1, 1),  # Stride rate of convolutional kernels.
    'conv_dilation_rate': (1, 1),  # Dilation rate of convolutional kernels.
    'conv_padding': (1, 1),  # Number of padded pixels in convolutional layers.
    'conv_padding_style': 'zeros',  # Style of padding.
    'n_classes': 1 ,# Number of output channels
    'input_channels': 1 # config['DataLoader']['Input_channel'] # we have two gray scale thingy, wpx and wpy (instead of RGB being channel 3) # now 9 :D
    }

VAE_Model = VUDNet.VAE(config['DataLoader']['Input_channel']).to(device)
UNet_Model = VUDNet.UNet(Unet_option).to(device)


unet_criterion = nn.MSELoss()

def VAE_loss_function(recon_x, x, mu, logvar):
        MSE = F.mse_loss(recon_x.squeeze(), x, reduction='sum')
        KLD = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
        return MSE + KLD


def combined_loss(outputs_unet, depth_maps, outputs_vae, mu, logvar):
    loss_unet = unet_criterion(outputs_unet, depth_maps)
    loss_vae = VAE_loss_function(outputs_vae, depth_maps, mu, logvar)
    return loss_unet + loss_vae

VAE_Model_optimizer = optim.Adam(VAE_Model.parameters(), lr=VAE_lr)
UNet_Model_optimizer = optim.Adam(UNet_Model.parameters(), lr=UNET_lr)


VAE_Model.train()
UNet_Model.train()
for epoch in range(num_epochs):
    running_loss = 0.0
    with tqdm(train_loader, desc=f'Epoch {epoch+1}/{num_epochs}', unit='batch') as t:
         for i, (images, depth_maps,_) in enumerate(t):
                images, depth_maps = images.to(device), depth_maps.to(device).float()

                VAE_Model_optimizer.zero_grad()
                UNet_Model_optimizer.zero_grad()
                
                # import pdb
                # pdb.set_trace()

                outputs_unet = UNet_Model(images)
                outputs_unet = outputs_unet.squeeze(1)
                recon_depth, mu, logvar = VAE_Model(images)
                recon_depth = recon_depth.squeeze(1)

                loss = combined_loss(outputs_unet, depth_maps, recon_depth,mu, logvar)
                loss.backward()

                VAE_Model_optimizer.step()
                UNet_Model_optimizer.step()

                running_loss += loss.item()
                # writer.add_scalar('training loss', running_loss / (i+1), epoch * len(train_loader) + i)
    print(f'Epoch {epoch + 1}, Loss: {running_loss / len(train_loader)}')               

fusion_model = VUDNet.FusionNet().to(device)
optimizer_fusion = optim.Adam(fusion_model.parameters(), lr=fusion_lr)
fusion_criterion = nn.MSELoss()
fusion_model.train()
VAE_Model.eval()
UNet_Model.eval()
for epoch in range(fusion_epochs):
    with tqdm(train_loader, desc=f'Epoch {epoch+1}/{fusion_epochs}', unit='batch') as t:
         for i, (images, depth_maps,_) in enumerate(t):
            images, depth_maps = images.to(device), depth_maps.to(device).float()

            with torch.no_grad():
                outputs_unet = UNet_Model(images)
                recon_depth, mu, logvar  = VAE_Model(images)
            inputs_fusion = torch.cat([outputs_unet, recon_depth], dim=1)
            optimizer_fusion.zero_grad()
            outputs = fusion_model(inputs_fusion)
            outputs = outputs.squeeze()
            loss = fusion_criterion(outputs, depth_maps)
            loss.backward()
            optimizer_fusion.step()
print('Finished Training')

if VISUALIZE_THE_DIST:
    visualize_latent_space(VAE_Model,train_loader,device)

fusion_model.eval()
VAE_Model.eval()
UNet_Model.eval()


test_loss = 0.0
with torch.no_grad():
    test_metrics = {'MSE': [], 'MAE': [], 'RMSE': [],  'LogError': []}
    for images_test, depth_maps_test,identifiers in test_loader:
        images_test, depth_maps_test = images_test.to(device), depth_maps_test.to(device).float()

        recon_depth,_, _ = VAE_Model(images_test)
        outputs_U = UNet_Model(images_test)

        inputs_fusion = torch.cat([outputs_U,recon_depth], dim=1)
        outputs_test = fusion_model(inputs_fusion)
        
        loss_test = fusion_criterion(outputs_test, depth_maps_test)
        test_loss += loss_test.item()

        if Show_PLOT_RESULTS:
            images_test_np = images_test.cpu().numpy()
            depth_maps_test_np = depth_maps_test.cpu().numpy()
            outputs_test_np = outputs_test.cpu().numpy()
            
            for i in range(len(images_test_np)):
                plot_depth_map_3d(depth_maps_test_np[i].squeeze())
                plot_depth_map_3d(outputs_test_np[i].squeeze())

        if CALCULATE_THE_CRITERIA:
            recon_depth_np = recon_depth.cpu().numpy()
            depth_maps_test_np = depth_maps_test.cpu().numpy()

            metrics = calculate_metrics(recon_depth_np, depth_maps_test_np)
            test_metrics['MSE'].append(metrics[0])
            test_metrics['MAE'].append(metrics[1])
            test_metrics['RMSE'].append(metrics[2])
            test_metrics['LogError'].append(metrics[3])

            if SAVE_DEPTH_MAP_RESULTS:
                for j, identifier in enumerate(identifiers):  # Iterate through batch
                    file_name = f'PDVUD{identifier}.npy'
                    file_path = os.path.join(NPY_SAVE_PATH, file_name)
                    np.save(file_path, recon_depth_np[j].squeeze())
                    
    if CALCULATE_THE_CRITERIA:
        for key in test_metrics:
            test_metrics[key] = np.mean(test_metrics[key])
        print (test_metrics)
print(f'Test Loss: {test_loss / len(test_loader)}')

