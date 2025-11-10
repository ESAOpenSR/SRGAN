import torch
import os
import einops
import rasterio
from rasterio.windows import Window
from torchvision import transforms
import random
import json
import numpy as np


class Sentinel2TestDataSet(torch.utils.data.Dataset):
    def __init__(self,data_folders = None,amount=100,band_selection="R10m"):
        # settings for band selection
        assert band_selection in ["R10m","R20m"]
        self.band_selection = band_selection
        self.amount = amount

        # def window return size
        self.image_size=128
        
        self.windows = []
        if type(data_folders) == type(None) or len(data_folders) == 0:
            # list files in default folder
            root_dir = "/data2/simon/test_s2/urban_tiles/"
            data_folders = [os.path.join(root_dir,folder) for folder in os.listdir(root_dir) if folder.endswith(".SAFE")]
            random.shuffle(data_folders)
            data_folders = data_folders[:5] # use only first 5 files for
        if type(data_folders) == str:
            self.get_windows(data_folders)
        if type(data_folders) == list:
            self.amount=int(self.amount/len(data_folders)) # divide by number of folders to keep total amount the same
            for data_folder_i in data_folders:
                self.get_windows(data_folder_i)
        
    def get_windows(self,data_folder):
    # get location of image data
        for dirpath, dirnames, _ in os.walk(data_folder):
            if "IMG_DATA" in dirnames:
                folder_path = os.path.join(dirpath, "IMG_DATA")
        folder_path = os.path.join(folder_path,self.band_selection)
        file_paths = os.listdir(folder_path)

        # get image file paths for selected bands
        if self.band_selection == "R10m":
            image_files = {"R":os.path.join(folder_path,[file for file in file_paths if "B04" in file][0]),
                        "G":os.path.join(folder_path,[file for file in file_paths if "B03" in file][0]),
                        "B":os.path.join(folder_path,[file for file in file_paths if "B02" in file][0]),
                        "NIR":os.path.join(folder_path,[file for file in file_paths if "B08" in file][0])}
        if self.band_selection == "R20m":
            image_files = {"B05":os.path.join(folder_path,[file for file in file_paths if "B05" in file][0]),
                        "B06":os.path.join(folder_path,[file for file in file_paths if "B06" in file][0]),
                        "B07":os.path.join(folder_path,[file for file in file_paths if "B07" in file][0]),
                        "B8A":os.path.join(folder_path,[file for file in file_paths if "B8A" in file][0]),
                        "B11":os.path.join(folder_path,[file for file in file_paths if "B11" in file][0]),
                        "B12":os.path.join(folder_path,[file for file in file_paths if "B12" in file][0])}
        
        # extract keys from image files
        band_names = list(image_files.keys())
        self.band_names = band_names
        
        # get iamge shape
        with rasterio.open(image_files[band_names[0]]) as src:
            image_width = src.width
            image_height = src.height
        
        # create list of coordinates
        for i in range(self.amount):
            rand_x = random.randint(0 ,image_width -self.image_size)
            rand_y = random.randint(0,image_height-self.image_size)
            window_ = Window(rand_x, rand_y, self.image_size, self.image_size)
            info_dict = {"path":image_files,"window":window_}
            self.windows.append(info_dict)
        random.shuffle(self.windows)

    def __len__(self):
        return len(self.windows)
    
    def __getitem__(self,idx):
        # get current window
        window = self.windows[idx]
        
        # read bands iteratively
        image=[]
        for band in self.band_names:
            band_file_path = window["path"][band]
            with rasterio.open(band_file_path) as src:
                window_data = src.read(1, window=window["window"])                
                image.append(window_data)
        image = np.stack(image)
        image = image/10000.0  # scale to [0,1]
        image = torch.Tensor(image).float()
        return(image)

if __name__ == "__main__":
    # Test DL
    dfs = None
    ds = Sentinel2TestDataSet(data_folders=dfs,band_selection="R10m")
    im = ds.__getitem__(20)

