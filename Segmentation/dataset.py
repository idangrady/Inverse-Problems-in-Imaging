import numpy as np
import torch
import torch.nn as nn
import os
from PIL import Image
from torch.utils.data import Dataset


class Dataset(Dataset):
    def __init__(self, image_dir, mask_dir, transform = None):
    
        self.image_dir = image_dir
        self.mask_dir =mask_dir
        self.transform = transform
        self.imgs = os.listdir(image_dir)
        
    def __len__(self):
        return len(self.imgs)
    
    def __getitem__(self, idx):
        img_path = os.path.join(self.image_dir, self.imgs[idx])
        mask_path =os.path.join(self.mask_dir, self.imgs[idx]) #TODO: check if naming are the same and adapt to it

        image = np.array(Image.open(img_path).convert("RGB"))#Convert to RGB
        mask = np.array(Image.open(mask_path).convert("L"),dtype=np.float32)

        #Convert Mask from 255 to 1
        mask = mask/255                                         #TODO: check if this works or you should do 255=1

        if self.transform is not None:
            augment =self.transform(image=image, mask = mask)
            img = augment["image"]
            mask = augment["mask"]

        #return
        return image, mask





        