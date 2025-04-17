import os
import random
import numpy as np
import torch
import torchvision.transforms as transforms
from torch.utils.data import Dataset, DataLoader
from PIL import Image
from scipy.ndimage import zoom
from glob import glob
from torchvision.transforms import InterpolationMode

class SegmentationDataset(Dataset):
    def __init__(self, root_dir='/path/to/SAR-dataset', subset='Train', low_res=None, transform_img=None, transform_mask=None, istrain=False):
        
        self.subset = subset
        self.root_dir = os.path.join(root_dir, subset)
        
        # Get all video folders
        self.video_dirs = []
        for d in os.listdir(self.root_dir):
            video_path = os.path.join(self.root_dir, d)
            if os.path.isdir(video_path) and os.path.exists(os.path.join(video_path, 'images')) and os.path.exists(os.path.join(video_path, 'segmentation')):
                self.video_dirs.append(d)
        
        # Constructing a list of image and mask paths
        self.img_path_all = []
        self.mask_path_all = []
        
        for video_dir in self.video_dirs:
            video_path = os.path.join(self.root_dir, video_dir)
            rgb_dir = os.path.join(video_path, 'images')
            seg_dir = os.path.join(video_path, 'segmentation')
            
            # Get all PNG files in the segmentation directory
            seg_files = sorted(glob(os.path.join(seg_dir, '*.png')))
            
            for seg_file in seg_files:
                # Get the corresponding RGB image file name from the split file name.
                frame_num = os.path.basename(seg_file)
                rgb_file = os.path.join(rgb_dir, frame_num)
                
                # Make sure the RGB file exists
                if os.path.exists(rgb_file):
                    self.img_path_all.append(rgb_file)
                    self.mask_path_all.append(seg_file)
        
        assert len(self.img_path_all) == len(self.mask_path_all), "Mismatch between the number of images and masks."
        
        print(f"Find {len(self.img_path_all)} pairs of Image and Segmentation Mask")
        
        self.transform_img = transform_img
        self.transform_mask = transform_mask
        self.istrain = istrain
        self.low_res = low_res
        
        # Data augment Parameters
        self.brightness = 0.1
        self.contrast = 0.1
        self.saturation = 0.1
        self.hue = 0.1
        self.color_aug = transforms.ColorJitter(self.brightness, self.contrast, self.saturation, self.hue)

    def __len__(self):
        return len(self.img_path_all)

    def __getitem__(self, idx):
        img_path = self.img_path_all[idx]
        mask_path = self.mask_path_all[idx]
        
        image = Image.open(img_path).convert('RGB')
        mask = Image.open(mask_path).convert('L')
        mask = np.array(mask)
        
        mask = Image.fromarray(mask)
        
        # Application data enhancement and conversion
        if self.istrain:
            # Random Horizontal Flip
            hflip = random.random() < 0.5
            flip_container = random.choice([Image.FLIP_LEFT_RIGHT, Image.FLIP_TOP_BOTTOM])
            if hflip:
                image = self.color_aug(image)
                image = image.transpose(flip_container)
                mask = mask.transpose(flip_container)
        
        if self.transform_img:
            image = self.transform_img(image)
        
        if self.transform_mask:
            mask = self.transform_mask(mask)
        
        # Convert to tensor
        mask = torch.from_numpy(np.array(mask)).long()
        sample = {'image': image, 'mask': mask}
        
        # If low resolution labels are required
        if self.low_res:
            low_res_label = zoom(np.array(mask), (self.low_res/np.array(mask).shape[0], self.low_res/np.array(mask).shape[1]), order=0)
            low_res_label = torch.from_numpy(low_res_label).long()
            sample = {'image': image, 'mask': mask, 'low_res_label': low_res_label}
        
        return sample

def get_loaders(args):
    # Define image and mask conversions
    transform_img = transforms.Compose([
        transforms.Resize((512, 512)),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225]
        )
    ])

    transform_mask = transforms.Compose([
        transforms.Resize((512, 512), interpolation=InterpolationMode.NEAREST),
    ])

    # Creating dataset instances
    train_dataset = SegmentationDataset(
        root_dir=args.data_path,
        subset='Train',
        low_res=128,
        transform_img=transform_img,
        transform_mask=transform_mask,
        istrain=True
    )
    
    test_dataset = SegmentationDataset(
        root_dir=args.data_path,
        subset='Validate',
        low_res=128,
        transform_img=transform_img,
        transform_mask=transform_mask
    )
    
    # Creating Data Loaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=2,
        pin_memory=True
    )
    
    test_loader = DataLoader(
        test_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=2,
        pin_memory=True
    )
    
    return train_loader, test_loader, train_dataset, test_dataset