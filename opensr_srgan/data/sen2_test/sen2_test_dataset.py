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
    
    """
    PyTorch Dataset for sampling small patches from Sentinel-2 SAFE products.

    This dataset reads Sentinel-2 imagery directly from original `.SAFE` folders
    as downloaded from Copernicus / ESA (no preprocessing required). It supports
    two predefined band selections:

    - ``band_selection="R10m"``: RGB + NIR at 10 m (B02, B03, B04, B08)
    - ``band_selection="R20m"``: 20 m bands (B05, B06, B07, B8A, B11, B12)

    For each SAFE folder, the dataset:
    1) Locates the ``IMG_DATA`` directory.
    2) Enters the corresponding resolution subfolder (``R10m`` or ``R20m``).
    3) Finds the required band files by band ID in the filename.
    4) Samples random ``image_size × image_size`` windows and stacks the
       selected bands into a tensor.

    Intensities are scaled by ``1/10000`` so that values are roughly in [0, 1].

    Parameters
    ----------
    data_folders : str or list of str or None, optional
        Path(s) to Sentinel-2 `.SAFE` folders. Each entry must be the root of
        a SAFE product (e.g. ``S2A_MSIL2A_20200101T105031_N0214_R051_T31SCA_20200101T125640.SAFE``).
        If:

        - ``None`` (default) or empty: the dataset scans the hard-coded directory
          ``/data2/simon/test_s2/urban_tiles/`` and uses up to 5 `.SAFE` folders
          found there.
        - ``str``: a single SAFE folder is used.
        - ``list``: each element is treated as a SAFE folder; the requested
          ``amount`` of samples is split across them.

        No preprocessed GeoTIFFs are required; the original ESA SAFE structure
        is expected on disk.

    amount : int, optional
        Approximate total number of random patches to sample across all SAFE
        folders. When multiple folders are provided as a list, this total is
        divided evenly across them. Default is ``100``.

    band_selection : {"R10m", "R20m"}, optional
        Predefined band group to load:

        - ``"R10m"``: returns a 4-channel tensor with keys
          ``["R", "G", "B", "NIR"]`` corresponding to
          ``B04``, ``B03``, ``B02``, ``B08``
        - ``"R20m"``: returns a 6-channel tensor with keys
          ``["B05", "B06", "B07", "B8A", "B11", "B12"]``

        Default is ``"R10m"``.

    Attributes
    ----------
    band_selection : str
        The selected band configuration (``"R10m"`` or ``"R20m"``).
    amount : int
        Number of patches per SAFE (after possible redistribution if multiple
        SAFE folders are passed).
    image_size : int
        Size of each square patch in pixels (fixed to 128).
    windows : list of dict
        Internal list of dictionaries with keys:
        - ``"path"``: mapping band-name → file path
        - ``"window"``: ``rasterio.windows.Window`` defining the patch.
    band_names : list of str
        List of band keys in the order they are stacked into the output tensor.

    Notes
    -----
    - The only required input is the path to one or more **raw Sentinel-2
      SAFE directories**. The dataset will locate the IMG_DATA/R10m or
      IMG_DATA/R20m subfolders and band files automatically.
    - Values are scaled by 10000 under the assumption of Sentinel-2 L1C/L2A
      reflectance conventions.

    Examples
    --------
    Basic usage with automatic discovery (hard-coded test path)::

        from torch.utils.data import DataLoader

        ds = Sentinel2TestDataSet(
            data_folders=None,
            amount=100,
            band_selection="R10m"
        )
        dl = DataLoader(ds, batch_size=4, shuffle=True)
        batch = next(iter(dl))   # shape: [4, 4, 128, 128] for R10m

    Using your own SAFE products::

        safe_paths = [
            "/path/to/S2A_..._T31SCA.SAFE",
            "/path/to/S2B_..._T31SCA.SAFE",
        ]
        ds = Sentinel2TestDataSet(
            data_folders=safe_paths,
            amount=200,
            band_selection="R20m"
        )
        patch = ds[0]  # shape: [6, 128, 128]
    """
    
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
        """
        Load a single multi-band patch.

        Parameters
        ----------
        idx : int
            Index of the patch to retrieve.

        Returns
        -------
        torch.FloatTensor
            Tensor of shape ``[C, H, W]`` where:

            - ``C`` is the number of bands (4 for R10m, 6 for R20m),
            - ``H = W = image_size`` (128 by default).

            Values are scaled by ``1/10000.0``.
        """
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

