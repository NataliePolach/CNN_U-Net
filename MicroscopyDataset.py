#!/usr/bin/env python3

# %%
# Loader for data
import os
import numpy as np
from PIL import Image
from torch.utils.data import Dataset


class MicroscopyDataset (Dataset):
    
    """Loading data fuction"""
    
    def __init__(self, image_dir, mask_dir, transform=None):
        
        #print("Path to image:" + str(image_dir))
        #print("Path to mask:" + str(mask_dir))
        self.image_dir = image_dir
        self.mask_dir = mask_dir   
        #list of all files in folder
        self.transform = transform
        self.images = os.listdir(image_dir)
        self.images = [ f for f in self.images if os.path.isfile(os.path.join(self.image_dir, f))]
    
    def __len__(self):
        
        #Length of the dataset
        return len(self.images)
    
    def __getitem__(self,idx):
        
        #Path for image and mask
        image_path = os.path.join(self.image_dir, self.images[idx])
        mask_path = os.path.join(self.mask_dir, self.images[idx])
        
        print(str(image_path))
        print(str(mask_path))
        
        image = np.array(Image.open(image_path).convert("L"))
        mask = np.array(Image.open(mask_path).convert("L"), dtype=np.float32)
        image = image.reshape(1, 256, 256).astype('float32') 
        mask = mask.reshape(1, 256, 256).astype('float32')
        mask[mask == 255.0] = 1.0
        return image, mask
        

if __name__ == "__main__":
    mds = MicroscopyDataset("train_img", "train_mask")
    for i in range(0, len(mds)):
        print(mds[i])
