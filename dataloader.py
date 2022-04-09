import os
from typing import Tuple
import cv2
import numpy as np
from torch.utils.data import Dataset, DataLoader
import torch
import torchvision
import torchvision.transforms as transforms
import argparse
from tqdm import tqdm
from typing import Tuple


def arg():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir", type = str, help = "data directory")
    return parser.parse_args()

# GAN is unsupervised learning 
# Thus, label ( target ) is not needed
# Hence, below dataset inputs and outputs only images.
# Image data shape is ( 3, 218, 178 )

# N = transforms.Normalize( (0.5, 0.5, 0.5), (0.5, 0.5, 0.5), inplace=False)

class custom_dataset(Dataset):
    def __init__(self, input_dir, transform = None):   # trasnformations : List
        self.input_dir = input_dir
        self.input_list = os.listdir(input_dir)
        self.transform = transform

    def __len__(self):
        return len(self.input_list)

    def __getitem__(self, idx):
        os.chdir(self.input_dir)
        input_image_numpy = cv2.imread(self.input_list[idx])     #  JPG -> numpy
        
        if self.transform != None:    # many transformations
            for tranform in self.transform:
                input_image_numpy = tranform(input_image_numpy)
        
        input_image_tensor = torchvision.transforms.functional.to_tensor(input_image_numpy)
        #input_image_tensor = N(input_image_tensor)
        return input_image_tensor

# Especially for face images, vetical flip could make strange face images.
# It's simple, for training dataset if upside down face added, model can make upsidedown images.
# Likewise, ratation not recommended. 

class RandomFlip():
    def __init__(self, horizontal = True, vertical = False, p = .5):
        self.horizontal = horizontal
        self.vertical = vertical
        self.p = p

    def __call__(self, face):
        if (self.horizontal) and (np.random.rand() > self.p):
            face = cv2.flip(face,1)
            
        if (self.vertical) and (np.random.rand() > self.p):
            face = cv2.flip(face,0)

        return face

class Resize():
    def __init__(self, size : Tuple[int, int]):
        self.size = size

    def __call__(self, face):
        face = cv2.resize(face, dsize = self.size, interpolation=cv2.INTER_LINEAR)

        return face

class Centercrop():
    def __init__(self, size : Tuple[int, int]):  # height, width
        self.size = size

    def __call__(self,face):
        height, width = face.shape[0], face.shape[1]    # grid is start form left above

        mid_x, mid_y = int(width/2), int(height/2)

        face = face[ mid_y-int(self.size[0]/2):mid_y+int(self.size[0]/2) , 
        mid_x-int(self.size[1]/2):mid_x+int(self.size[1]/2),
         : ]

        return face


# It's GAN therefore test dataset is not needed.

def make_dataloader(train_dir, batch_size, transform = [Centercrop((160,160)), Resize((64,64)), RandomFlip()]):    # For many transformation, type should be list

    train_dataset = custom_dataset(train_dir, transform)
    train_dl = DataLoader(train_dataset, batch_size, shuffle = True, num_workers = 4, pin_memory = True)

    return train_dl


''' 🤓🤓🤓🤓🤓🤓🤓🤓🤓🤓🤓🤓🤓🤓🤓🤓
def make_dataloader(train_dir, batch_size, transform = [Centercrop((160,160)), RandomFlip()]):    # For many transformation, type should be list

    train_dataset = custom_dataset(train_dir, transform)
    train_dl = DataLoader(train_dataset, batch_size, shuffle = True, num_workers = 4, pin_memory = True)
🤓🤓🤓🤓🤓🤓🤓🤓🤓🤓🤓🤓🤓🤓🤓🤓 ''' 


    return train_dl

# Image dataset size distribution

def check_size(data_dir):
    image_size_dict = dict()
    image_size_set = set()
    os.chdir(data_dir)
    data_list = os.listdir(data_dir)

    for image_name in tqdm(data_list):
        image_size = cv2.imread(image_name).shape  # numpy.shape
        if not image_size in image_size_set:
            image_size_dict[image_size] = 0
            image_size_set.add(image_size)
        image_size_dict[image_size] += 1

    print(image_size_dict)





if __name__ == "__main__":
    arg = arg()
    data_dir = arg.data_dir
    print("calculating image size distribution .............................................")
    
    check_size(data_dir)

    print("dataset test .............................................")

    train_dataset = custom_dataset(data_dir)

    for i in train_dataset:
        print("len of train dataset", len(train_dataset))
        print("shape and target of train dataset ",i.shape,)
        break
    print("dataloader test .............................................")

    train_dl= make_dataloader(data_dir, batch_size = 16) 

    for i in train_dl:
        print("epochs train", len(train_dl))
        print("shape and target of train data ",i.shape)
        break