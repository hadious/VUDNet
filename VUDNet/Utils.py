from keras import backend as Kend
from keras.models import load_model
from Surface_VUD import Surface_VUD
import torch
from torch.utils.data import random_split
import numpy as np 
import matplotlib.pyplot as plt
import os
import tensorflow as tf
from sklearn.manifold import TSNE
from matplotlib.offsetbox import OffsetImage, AnnotationBbox
from PIL import Image


def Load_the_dataset(config, transform):

    datset_path = config['dataset_path']
    image_suffix = config['image_suffix']
    depthMap_suffix = config['depthMap_suffix']
    batch_size =  config['batch_size']
    TrainSplit = config['TrainSplit']
    ValidationSplit = config['ValidationSplit']
    DYNETPath = config['DYNETPath']

    Kend.clear_session()
    DYModel = load_model(DYNETPath, custom_objects={"tf": tf})


    dataset = Surface_VUD(datset_path, image_suffix, depthMap_suffix, transform, DYModel)
    data_loader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=True)

    train_size = int(TrainSplit * len(dataset))   
    val_size = int(ValidationSplit * len(dataset))    
    test_size = len(dataset) - train_size - val_size   

    train_set, val_set, test_set = random_split(dataset, [train_size, val_size, test_size])

    train_loader = torch.utils.data.DataLoader(train_set, batch_size=batch_size, shuffle=True)
    val_loader = torch.utils.data.DataLoader(val_set, batch_size=batch_size, shuffle=False)
    test_loader = torch.utils.data.DataLoader(test_set, batch_size=batch_size, shuffle=False)

    return train_loader, val_loader, test_loader

def plot_depth_map_3d(depth_map):
        X, Y = np.meshgrid(np.arange(depth_map.shape[1]), np.arange(depth_map.shape[0]))


        crop_top = 20
        crop_bottom = 0
        crop_left = 0
        crop_right=0
        # import pdb
        # pdb.set_trace()
        depth_map = depth_map[crop_top:depth_map.shape[0]-crop_bottom, crop_left:depth_map.shape[1]-crop_right]
        X, Y = np.meshgrid(np.arange(depth_map.shape[1]), np.arange(depth_map.shape[0]))

        # Plot the depth map in 3D
        fig = plt.figure()
        ax = fig.add_subplot(111, projection='3d')
        ax.plot_surface(X, Y, depth_map, cmap='viridis')  
        ax.set_xlabel('Width')
        ax.set_ylabel('Height')
        ax.set_zlabel('Depth')
        ax.set_title('Depth Map Visualization')

        ax.set_zlim(0, 1) 
        plt.show()

def calculate_metrics(pred, truth):
    pred = pred.flatten()
    truth = truth.flatten()
    
    mse = ((pred - truth) ** 2).mean()
    mae = np.abs(pred - truth).mean()
    rmse = np.sqrt(mse)
    log_error = np.mean(np.abs(np.log10(pred + 1) - np.log10(truth + 1)))  # Added 1 to avoid log(0)
    
    return mse, mae, rmse, log_error

def visualize_latent_space(model, dataloader,device):
    model.eval()
    latent_vectors = []
    labels = []
    images = []
    
    with torch.no_grad():
        for data, label,_ in dataloader:
            data = data.to(device)
            _, mu, _ = model(data)
            # images.append(data.cpu().numpy())
            latent_vectors.append(mu.cpu().numpy())
            labels.append(label.numpy())
    
    latent_vectors = np.concatenate(latent_vectors, axis=0)
    labels = np.concatenate(labels, axis=0)


    tsne = TSNE(n_components=2, random_state=35)
    tsne_results = tsne.fit_transform(latent_vectors)
    
    plt.figure(figsize=(10, 6))
    plt.scatter(tsne_results[:, 0], tsne_results[:, 1], cmap='viridis')
    plt.colorbar()
    plt.title('t-SNE of VAE Latent Space')
    plt.xlabel('t-SNE 1')
    plt.ylabel('t-SNE 2')
    plt.show()

    # return images,labels


def visualize_latent_space_with_image(model, dataloader, device):
    model.eval()
    latent_vectors = []
    images = []
    
    with torch.no_grad():
        for data, label,_ in dataloader:
            images.append(data.numpy())
            data = data.to(device)
            _, mu, _ = model(data)
            latent_vectors.append(mu.cpu().numpy())
    
    latent_vectors = np.concatenate(latent_vectors, axis=0)
    images = np.concatenate(images, axis=0)

    tsne = TSNE(n_components=2, random_state=35)
    tsne_results = tsne.fit_transform(latent_vectors)
    
    plt.figure(figsize=(14, 10))
    scatter = plt.scatter(tsne_results[:, 0], tsne_results[:, 1], cmap='viridis')
    
    num_images_to_plot = 4
    np.random.seed(35) 
    indices_to_plot = np.random.choice(range(tsne_results.shape[0]), size=num_images_to_plot, replace=False)
    for i in indices_to_plot:
        image = images[i]
        image = np.transpose(image, (1, 2, 0))  
        imagebox = OffsetImage(image, zoom=0.3, alpha=0.6)
        ab = AnnotationBbox(imagebox, (tsne_results[i, 0], tsne_results[i, 1]), frameon=False)
        plt.gca().add_artist(ab)
    
    plt.colorbar(scatter)
    plt.title('t-SNE of VAE Latent Space')
    plt.xlabel('t-SNE 1')   
    plt.ylabel('t-SNE 2')
    plt.show()



def visualize_latent_space_with_number(model, dataloader, device):
    model.eval()
    latent_vectors = []
    images = []


    with torch.no_grad():
        for data, _, img_name in dataloader:
            images.append(data.numpy())
            
            data = data.to(device)
            _, mu, _ = model(data)
            latent_vectors.append(mu.cpu().numpy())
    
    latent_vectors = np.concatenate(latent_vectors, axis=0)
    images = np.concatenate(images, axis=0)

    tsne = TSNE(n_components=2, random_state=35)
    tsne_results = tsne.fit_transform(latent_vectors)
    
    plt.figure(figsize=(14, 10))
    scatter = plt.scatter(tsne_results[:, 0], tsne_results[:, 1], cmap='viridis')
    
    num_images_to_plot = 12
    np.random.seed(35) 
    indices_to_plot = np.random.choice(range(tsne_results.shape[0]), size=num_images_to_plot, replace=False)


    for i, idx in enumerate(indices_to_plot):
        plt.annotate(str(i+1), (tsne_results[idx, 0], tsne_results[idx, 1]), fontsize=12, fontweight='bold', color='red')

    plt.colorbar(scatter)
    plt.title('t-SNE of VAE Latent Space')
    plt.xlabel('t-SNE 1')   
    plt.ylabel('t-SNE 2')
    plt.show()


    
    

def visualize_latent_space_with_image_3d(model, dataloader, device):
    model.eval()
    latent_vectors = []
    images = []
    
    with torch.no_grad():
        for data, label, _ in dataloader:
            images.append(data.numpy())
            data = data.to(device)
            _, mu, _ = model(data)
            latent_vectors.append(mu.cpu().numpy())
    
    latent_vectors = np.concatenate(latent_vectors, axis=0)
    images = np.concatenate(images, axis=0)

    tsne = TSNE(n_components=3, random_state=35)
    tsne_results = tsne.fit_transform(latent_vectors)
    
    fig = plt.figure(figsize=(14, 10))
    ax = fig.add_subplot(111, projection='3d')
    
    scatter = ax.scatter(tsne_results[:, 0], tsne_results[:, 1], tsne_results[:, 2], cmap='viridis')
    
    num_images_to_plot = 0
    np.random.seed(35)
    indices_to_plot = np.random.choice(range(tsne_results.shape[0]), size=num_images_to_plot, replace=False)
    indices_to_plot = [1, 7, 9, 12, 18]
    for i in indices_to_plot:
        image = images[i]
        image = np.transpose(image, (1, 2, 0))  # Assuming the image is in CHW format
        imagebox = OffsetImage(image, zoom=0.3, alpha=0.6)
        ab = AnnotationBbox(imagebox, (tsne_results[i, 0], tsne_results[i, 1]), frameon=False)
        ax.add_artist(ab)
    
    fig.colorbar(scatter)
    ax.set_title('3D t-SNE of VAE Latent Space')
    ax.set_xlabel('t-SNE 1')
    ax.set_ylabel('t-SNE 2')
    ax.set_zlabel('t-SNE 3')
    plt.show()


def visualize_latent_space_with_image_sperate(model, dataloader, device):
    model.eval()
    latent_vectors = []
    images = []
    
    with torch.no_grad():
        for data, label, _ in dataloader:
            images.append(data.numpy())
            data = data.to(device)
            _, mu, _ = model(data)
            latent_vectors.append(mu.cpu().numpy())
    
    latent_vectors = np.concatenate(latent_vectors, axis=0)
    images = np.concatenate(images, axis=0)

    tsne = TSNE(n_components=2, random_state=35)
    tsne_results = tsne.fit_transform(latent_vectors)
    
    plt.figure(figsize=(14, 10))
    scatter = plt.scatter(tsne_results[:, 0], tsne_results[:, 1], cmap='viridis')
    
    num_images_to_plot = 8
    np.random.seed(35) 
    indices_to_plot = np.random.choice(range(tsne_results.shape[0]), size=num_images_to_plot, replace=False)
     
    for i, idx in enumerate(indices_to_plot):
        plt.annotate(str(i+1), (tsne_results[idx, 0], tsne_results[idx, 1]), fontsize=12, fontweight='bold', color='red')

    fig, axes = plt.subplots(1, num_images_to_plot, figsize=(15, 4))
    for ax, i in zip(axes, indices_to_plot):
        image = images[i]
        image = np.transpose(image, (1, 2, 0))
        if image.shape[0] == 3:  
            image = np.dot(image.T, [0.2989, 0.5870, 0.1140]).T  # RGB to grayscale conversion
        ax.imshow(image, cmap='gray')
        ax.axis('off')
        ax.set_title(f'Index: {i}')

    plt.suptitle('Selected Images')
    plt.show()
    
    plt.figure(figsize=(14, 10))
    scatter = plt.scatter(tsne_results[:, 0], tsne_results[:, 1], cmap='viridis')
    plt.colorbar(scatter)
    plt.title('t-SNE of VAE Latent Space')
    plt.xlabel('t-SNE 1')   
    plt.ylabel('t-SNE 2')
    plt.show()