from torch.utils.data import Dataset
import os 
from PIL import Image
import numpy as np 

class Surface_Dataset(Dataset):

    def __init__(self, path,  image_suffix, depthMap_suffix, transform):
        self.path = path 
        self.image_suffix = image_suffix 
        self.depthMap_suffix = depthMap_suffix
        self.image_files = [f for f in os.listdir(path) if f.endswith(image_suffix)]
        self.transform = transform
        
    
    def __len__(self):
         return len(self.image_files)
    
    def __getitem__(self, index):
        image_name = self.image_files[index]
        image_path = os.path.join(self.path, image_name)
        depth_map_path = os.path.join(self.path, image_name.replace(self.image_suffix, self.depthMap_suffix))
        depth_map_path =  depth_map_path.replace("2d", "depth")
        
        # Load image
        image = Image.open(image_path).convert('RGB')
        image = self.transform(image)

        # Load depth map
        depth_data = np.load(depth_map_path)
        depthMap = depth_data['arr_0']
        normalized_depthmap = (depthMap - depthMap.mean()) / depthMap.std()
        
        
        return image, normalized_depthmap 
