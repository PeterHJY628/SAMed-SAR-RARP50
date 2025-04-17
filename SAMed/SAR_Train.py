import os
import cv2
import json
import random
import torch
import torchvision.transforms as transforms
from torch.utils.data import Dataset, DataLoader
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt
import imageio.v2 as iio
from torchvision.transforms import InterpolationMode
from scipy import ndimage
from scipy.ndimage import zoom
from glob import glob
from einops import repeat
#--------
import os
import cv2

import sys
from tqdm import tqdm
import logging
import numpy as np
import argparse
import random
import numpy as np
import torch
from torch.utils.data import DataLoader
import torch.backends.cudnn as cudnn
from importlib import import_module
from segment_anything import sam_model_registry
from datasets.dataset_synapse import Synapse_dataset
from icecream import ic
from medpy import metric
from scipy.ndimage import zoom
import torch.nn as nn
import SimpleITK as sitk
import torch.nn.functional as F
import imageio
from einops import repeat
from pathlib import Path
import monai
import warnings

from torch.nn.modules.loss import CrossEntropyLoss
from utils import DiceLoss
import torch.optim as optim
from collections import Counter

import matplotlib.pyplot as plt
#---------
from segment_anything import build_sam, SamPredictor
from segment_anything import sam_model_registry

import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
from torch.nn.parameter import Parameter
from segment_anything.modeling import Sam
from safetensors import safe_open
from safetensors.torch import save_file

from icecream import ic


def normalise_intensity(image, ROI_thres=0.1):
    pixel_thres = np.percentile(image, ROI_thres)
    ROI = np.where(image > pixel_thres, image, 0) # If image value is greater than pixel threshold, return image value, otherwise return 0
    mean = np.mean(ROI)
    std = np.std(ROI)
    ROI_norm = (ROI - mean) / (std + 1e-8) # Normalise ROI
    return ROI_norm

def random_rot_flip(image, label):
    k = np.random.randint(0, 4)
    image = np.rot90(image, k)
    label = np.rot90(label, k)
    axis = np.random.randint(0, 2)
    image = np.flip(image, axis=axis).copy()
    label = np.flip(label, axis=axis).copy()
    return image, label

def random_rotate(image, label):
    angle = np.random.randint(-20, 20)
    image = ndimage.rotate(image, angle, order=0, reshape=False)
    label = ndimage.rotate(label, angle, order=0, reshape=False)
    return image, label

class SegmentationDataset(Dataset):
    def __init__(self, root_dir='/path/to/SAR-dataset', subset='Train', low_res=None, transform_img=None, transform_mask=None, istrain=False):
        
        self.subset = subset
        self.root_dir = os.path.join(root_dir, subset)
        
        # # Get all video folders
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
        
        print(f"Find{len(self.img_path_all)}pars of Image and Segmentation Mask")
        
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
            low_res_label = zoom(mask, (self.low_res/mask.shape[0], self.low_res/mask.shape[1]), order=0)
            sample = {'image': image, 'mask': mask, 'low_res_label': low_res_label}
        
        return sample

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
    # transforms.ToTensor(),  # Already converted to a tensor in __getitem__
    ])

# Creating a dataset instance
# Please replace the path with the path to your SAR-RARP50 dataset
dataset = SegmentationDataset(
    root_dir=r'Your Path',
    subset='Train',
    transform_img=transform_img,
    transform_mask=transform_mask,
    istrain=True
)

# Creating a Data Loader
batch_size = 8
train_loader = DataLoader(
    dataset,
    batch_size=batch_size,
    shuffle=True,
    num_workers=4,
    pin_memory=True
)

#------------------------LoRA part------------------------

from segment_anything import build_sam, SamPredictor
from segment_anything import sam_model_registry

import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
from torch.nn.parameter import Parameter
from segment_anything.modeling import Sam
from safetensors import safe_open
from safetensors.torch import save_file

from icecream import ic

class _LoRA_qkv_v0_v2(nn.Module):

    def __init__(
            self,
            qkv: nn.Module,
            linear_a_q: nn.Module,
            linear_b_q: nn.Module,
            linear_a_v: nn.Module,
            linear_b_v: nn.Module,
            conv_se_q: nn.Module,
            conv_se_v: nn.Module,
    ):
        super().__init__()
        self.qkv = qkv
        self.linear_a_q = linear_a_q
        self.linear_b_q = linear_b_q
        self.linear_a_v = linear_a_v
        self.linear_b_v = linear_b_v
        self.conv_se_q = conv_se_q
        self.conv_se_v = conv_se_v

        self.dim = qkv.in_features
        self.w_identity = torch.eye(qkv.in_features)

    def forward(self, x):
        qkv = self.qkv(x)
        a_q_out = self.linear_a_q(x)
        a_v_out = self.linear_a_v(x)
        a_q_out_temp = self.conv_se_q(a_q_out.permute(0,3,1,2)).permute(0,2,3,1)
        a_v_out_temp = self.conv_se_v(a_v_out.permute(0,3,1,2)).permute(0,2,3,1)

        new_q = self.linear_b_q(torch.mul(a_q_out, torch.sigmoid(a_q_out_temp)))#SE = Squeeze and Excitation
        new_v = self.linear_b_v(torch.mul(a_v_out, torch.sigmoid(a_v_out_temp)))

        qkv[:, :, :, : self.dim] += new_q
        qkv[:, :, :, -self.dim:] += new_v
        return qkv

class LoRA_Sam_v0_v2(nn.Module):

    def __init__(self, sam_model: Sam, r: int, lora_layer=None):
        super(LoRA_Sam_v0_v2, self).__init__()

        assert r > 0
        if lora_layer:
            self.lora_layer = lora_layer
        else:
            self.lora_layer = list(
                range(len(sam_model.image_encoder.blocks)))  # Only apply lora to the image encoder by default
        # create for storage, then we can init them or load weights
        self.w_As = []  # These are linear layers
        self.w_Bs = []

        # lets freeze first
        for param in sam_model.image_encoder.parameters():
            param.requires_grad = False

        # Here, we do the surgery
        for t_layer_i, blk in enumerate(sam_model.image_encoder.blocks):
            # If we only want few lora layer instead of all
            if t_layer_i not in self.lora_layer:
                continue
            w_qkv_linear = blk.attn.qkv
            self.dim = w_qkv_linear.in_features
            w_a_linear_q = nn.Linear(self.dim, r, bias=False)
            w_b_linear_q = nn.Linear(r, self.dim, bias=False)
            w_a_linear_v = nn.Linear(self.dim, r, bias=False)
            w_b_linear_v = nn.Linear(r, self.dim, bias=False)

            conv_se_q = nn.Conv2d(r, r, kernel_size=1,
                                    stride=1, padding=0, bias=False)
            conv_se_v = nn.Conv2d(r, r, kernel_size=1,
                                    stride=1, padding=0, bias=False)
            self.w_As.append(w_a_linear_q)
            self.w_Bs.append(w_b_linear_q)
            self.w_As.append(w_a_linear_v)
            self.w_Bs.append(w_b_linear_v)
            self.w_As.append(conv_se_q)
            self.w_As.append(conv_se_v)
            blk.attn.qkv = _LoRA_qkv_v0_v2(
                w_qkv_linear,
                w_a_linear_q,
                w_b_linear_q,
                w_a_linear_v,
                w_b_linear_v,
                conv_se_q,
                conv_se_v,
            )
        self.reset_parameters()
        self.sam = sam_model

    def save_lora_parameters(self, filename: str) -> None:
        r"""Only safetensors is supported now.

        pip install safetensor if you do not have one installed yet.

        save both lora and fc parameters.
        """

        assert filename.endswith(".pt") or filename.endswith('.pth')

        num_layer = len(self.w_As)  # actually, it is half
        a_tensors = {f"w_a_{i:03d}": self.w_As[i].weight for i in range(num_layer)}
        num_layer = len(self.w_Bs)  # actually, it is half
        b_tensors = {f"w_b_{i:03d}": self.w_Bs[i].weight for i in range(num_layer)}
        prompt_encoder_tensors = {}
        mask_decoder_tensors = {}

        # save prompt encoder, only `state_dict`, the `named_parameter` is not permitted
        if isinstance(self.sam, torch.nn.DataParallel) or isinstance(self.sam, torch.nn.parallel.DistributedDataParallel):
            state_dict = self.sam.module.state_dict()
        else:
            state_dict = self.sam.state_dict()
        for key, value in state_dict.items():
            if 'prompt_encoder' in key:
                prompt_encoder_tensors[key] = value
            if 'mask_decoder' in key:
                mask_decoder_tensors[key] = value

        merged_dict = {**a_tensors, **b_tensors, **prompt_encoder_tensors, **mask_decoder_tensors}
        torch.save(merged_dict, filename)

    def load_lora_parameters(self, filename: str) -> None:
        r"""Only safetensors is supported now.

        pip install safetensor if you do not have one installed yet.\

        load both lora and fc parameters.
        """

        assert filename.endswith(".pt") or filename.endswith('.pth')

        state_dict = torch.load(filename, weights_only=False)

        for i, w_A_linear in enumerate(self.w_As):
            saved_key = f"w_a_{i:03d}"
            # print('mobarak:', saved_key)
            saved_tensor = state_dict[saved_key]
            w_A_linear.weight = Parameter(saved_tensor)

        for i, w_B_linear in enumerate(self.w_Bs):
            saved_key = f"w_b_{i:03d}"
            saved_tensor = state_dict[saved_key]
            w_B_linear.weight = Parameter(saved_tensor)

        sam_dict = self.sam.state_dict()
        sam_keys = sam_dict.keys()

        # load prompt encoder
        prompt_encoder_keys = [k for k in sam_keys if 'prompt_encoder' in k]
        prompt_encoder_values = [state_dict[k] for k in prompt_encoder_keys]
        prompt_encoder_new_state_dict = {k: v for k, v in zip(prompt_encoder_keys, prompt_encoder_values)}
        sam_dict.update(prompt_encoder_new_state_dict)

        # load mask decoder
        mask_decoder_keys = [k for k in sam_keys if 'mask_decoder' in k]
        mask_decoder_values = [state_dict[k] for k in mask_decoder_keys]
        mask_decoder_new_state_dict = {k: v for k, v in zip(mask_decoder_keys, mask_decoder_values)}
        sam_dict.update(mask_decoder_new_state_dict)
        self.sam.load_state_dict(sam_dict)

    def reset_parameters(self) -> None:
        for w_A in self.w_As:
            nn.init.kaiming_uniform_(w_A.weight, a=math.sqrt(5))
        for w_B in self.w_Bs:
            nn.init.zeros_(w_B.weight)

    def forward(self, batched_input, multimask_output, image_size):
        return self.sam(batched_input, multimask_output, image_size)
    #------------------------LoRA part------------------------
    
    #------------------------Training & Validation part------------------------
import os
import cv2

import sys
from tqdm import tqdm
import logging
import numpy as np
import argparse
import random
import numpy as np
import torch
from torch.utils.data import DataLoader
import torch.backends.cudnn as cudnn
from importlib import import_module
from segment_anything import sam_model_registry
from datasets.dataset_synapse import Synapse_dataset
from icecream import ic
from medpy import metric
from scipy.ndimage import zoom
import torch.nn as nn
import SimpleITK as sitk
import torch.nn.functional as F
import imageio
from einops import repeat
from pathlib import Path
import monai
import warnings

from torch.nn.modules.loss import CrossEntropyLoss
from utils import DiceLoss
import torch.optim as optim
from collections import Counter

import matplotlib.pyplot as plt


def save_one_hot(root_dir, oh):
    # function to store a one hot tensor as separate images 
    for i, c in enumerate(oh):
        cv2.imwrite(f'{root_dir}/{i}.png', c.numpy().astype(np.uint8)*255)

def imread_one_hot(filepath, n_classes):
    # reads a segmentation mask stored as png and returns in in one-hot torch.tensor format
    img = cv2.imread(str(filepath))
    if img is None:
        raise FileNotFoundError(filepath)
    if len(img.shape)==3:  # if the segmentation mask was 3 channel, only keep the first
        img = img[...,0]
    img = torch.from_numpy(img).unsqueeze(0).unsqueeze(0).requires_grad_(False)
    return monai.networks.utils.one_hot(img, n_classes, dim=1)

def get_val_func(metric, n_classes=9, fix_nans=False):
    # this function is intended for wrapping the meanIoU and meanNSD metric computation functions
    # It returns a error computation function that is able to parse reference 
    # and prediction segmentation samples in directory level. 
    def f(dir_pred, dir_ref):
        seg_ref_paths = sorted(list(dir_ref.iterdir()))
        dir_pred = Path(dir_pred)
        
        acc=[]
        with torch.no_grad():
            for seg_ref_p in seg_ref_paths:
                # load segmentation masks as one_hot torch tensors
                try:
                    ref = imread_one_hot(seg_ref_p, n_classes=n_classes+1)
                except FileNotFoundError:
                    raise
                try:
                    pred = imread_one_hot(dir_pred/seg_ref_p.name, n_classes=n_classes+1)
                except FileNotFoundError as e:
                    # if the prediction file was not found, set all scores to zero and continue
                    acc.append([0]*n_classes) 
                    continue
                
                if fix_nans:
                    with warnings.catch_warnings():
                        warnings.simplefilter("ignore")
                        err = metric(pred, ref)
                    # if both reference and predictions are zero, set the prediction values to one
                    # this is required for NSD because otherwise values are going to
                    # be set to nan even though the prediction is correct.
                    # in case either the pred or corresponding ref channel is zero
                    # NSD will return either 0 or nan and in those cases nan is 
                    # converted to zero
                    # find the zero channels in both ref and pred and create a mask 
                    # in the size of the final prediction.(1xn_channels)
                    r_m, p_m = ref.mean(axis=(2,3)), pred.mean(axis=(2,3))
                    mask = ((r_m ==0) * (p_m == 0))[:,1:]

                    # set the scores in cases where both ref and pred were full zero
                    #  to 1.
                    err[mask==True]=1
                    # in cases where either was full zero but the other wasn't 
                    # (score is nan ) set the corresponding score to 0
                    err.nan_to_num_()
                else:
                    err = metric(pred, ref)
                acc.append(err.detach().reshape(-1).tolist())
        return np.array(acc)  # need to add axis and then multiply with the channel scales
    return f
    
def mIoU(dir_pred, dir_ref, n_classes=9):
    metric = monai.metrics.MeanIoU(include_background=False, reduction='mean', get_not_nans=False, ignore_empty=False)
    validation_func = get_val_func(metric, n_classes=n_classes)
    return validation_func(dir_pred/'segmentation', dir_ref/'segmentation')

def mNSD(dir_pred, dir_ref, n_classes=9, channel_tau=[1]*9):
    metric = monai.metrics.SurfaceDiceMetric(channel_tau,
                                            include_background=False,
                                            reduction='mean')
    validation_func = get_val_func(metric, n_classes=n_classes, fix_nans=True)
    return validation_func(dir_pred/'segmentation', dir_ref/'segmentation')


def adjust_learning_rate(optimizer, iter_num, args):
    if args.warmup and iter_num < args.warmup_period:
        lr_ = args.base_lr * ((iter_num + 1) / args.warmup_period)
    else:
        if args.warmup:
            shift_iter = iter_num - args.warmup_period
            assert shift_iter >= 0, f'Shift iter is {shift_iter}, smaller than zero'
        else:
            shift_iter = iter_num
        lr_ = args.base_lr * (1.0 - shift_iter / args.max_iterations) ** 0.9
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr_
    return lr_

def calculate_confusion_matrix_from_arrays(prediction, ground_truth, nr_labels):
    replace_indices = np.vstack((
        ground_truth.flatten(),
        prediction.flatten())
    ).T
    confusion_matrix, _ = np.histogramdd(
        replace_indices,
        bins=(nr_labels, nr_labels),
        range=[(0, nr_labels), (0, nr_labels)]
    )
    confusion_matrix = confusion_matrix.astype(np.uint32)
    return confusion_matrix

def calculate_dice(confusion_matrix):
    dices = []
    for index in range(confusion_matrix.shape[0]):
        true_positives = confusion_matrix[index, index]
        false_positives = confusion_matrix[:, index].sum() - true_positives
        false_negatives = confusion_matrix[index, :].sum() - true_positives
        denom = 2 * true_positives + false_positives + false_negatives
        if denom == 0:
            dice = 0
        else:
            dice = 2 * float(true_positives) / denom
        dices.append(dice)
    return dices

def inference_per_epoch(model, testloader, ce_loss, dice_loss, multimask_output=True, args=None, save_predictions=False):
    model.eval()
    # fig, axs = plt.subplots(len(testloader), 3, figsize=(1*3, len(testloader)*1), subplot_kw=dict(xticks=[],yticks=[]))
    loss_per_epoch, dice_per_epoch = [], []
    num_classes = args.num_classes + 1
    confusion_matrix = np.zeros((num_classes, num_classes), dtype=np.uint32)
    class_wise_dice = []
    
    # Creating inferencing and reference catalogues
    if save_predictions:
        prediction_dir = Path(args.output_dir) / "predictions"
        reference_dir = Path(args.output_dir) / "references"
        
        # Make sure the catalogue exists
        (prediction_dir / "segmentation").mkdir(parents=True, exist_ok=True)
        (reference_dir / "segmentation").mkdir(parents=True, exist_ok=True)
    
    with torch.no_grad():
        for i_batch, sampled_batch in enumerate(testloader):
            image_batch, label_batch, low_res_label_batch = sampled_batch['image'], sampled_batch['mask'], sampled_batch['low_res_label']
            image_batch, label_batch, low_res_label_batch = image_batch.to(device, dtype=torch.float32), label_batch.to(device, dtype=torch.long), low_res_label_batch.to(device, dtype=torch.long)
            outputs = model(image_batch, multimask_output, args.img_size)
            logits = outputs['masks']
            prob = F.softmax(logits, dim=1)
            pred_seg = torch.argmax(prob, dim=1)
            confusion_matrix += calculate_confusion_matrix_from_arrays(pred_seg.cpu().detach().numpy(), label_batch.cpu().detach().numpy(), num_classes)
            loss, loss_ce, loss_dice = calc_loss(logits, label_batch, ce_loss, dice_loss, args)
            loss_per_epoch.append(loss.item())
            dice_per_epoch.append(1-loss_dice.item())
            low_res_logits = outputs['low_res_logits']
            loss_dice = dice_loss(low_res_logits, low_res_label_batch, softmax=True)
            
            # Saving predictions and reference split masks
            if save_predictions:
                pred_cpu = pred_seg.cpu().numpy()
                label_cpu = label_batch.cpu().numpy()
                
                for j in range(pred_cpu.shape[0]):
                    img_idx = i_batch * args.batch_size + j
                    # Preservation of predicion and reference segmentation masks
                    cv2.imwrite(str(prediction_dir / "segmentation" / f"{img_idx:04d}.png"), pred_cpu[j].astype(np.uint8))
                    cv2.imwrite(str(reference_dir / "segmentation" / f"{img_idx:04d}.png"), label_cpu[j].astype(np.uint8))

        confusion_matrix = confusion_matrix[1:, 1:]  # exclude background
        dices_per_class = {'dice_cls:{}'.format(cls + 1): dice
                    for cls, dice in enumerate(calculate_dice(confusion_matrix))}
    

    official_metrics = None
    if save_predictions and args.use_official_metrics:
        try:
            miou_scores = mIoU(prediction_dir, reference_dir, n_classes=args.num_classes)
            mnsd_scores = mNSD(prediction_dir, reference_dir, n_classes=args.num_classes)
            
            # Calculation of average values
            mean_miou = np.mean(miou_scores)
            mean_mnsd = np.mean(mnsd_scores)
            
            # Calculating the final score
            final_score = np.sqrt(mean_miou * mean_mnsd)
            
            official_metrics = {
                'mIoU': mean_miou,
                'mNSD': mean_mnsd,
                'Final_Score': final_score
            }
            
            print(f"Official mIoU: {mean_miou:.4f}")
            print(f"Official mNSD: {mean_mnsd:.4f}")
            print(f"Official Final Score: {final_score:.4f}")
        except Exception as e:
            print(f"Error computing official metrics: {e}")

    return np.mean(loss_per_epoch), np.mean(dice_per_epoch), dices_per_class, official_metrics

def seed_everything(seed=42):
    cudnn.benchmark = False
    cudnn.deterministic = True
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)

def calc_loss(output, label_batch, ce_loss, dice_loss, args):
    # print("label_batch",label_batch.shape)
    # print(output.shape)
    loss_ce = ce_loss(output, label_batch[:].long())
    loss_dice = dice_loss(output, label_batch, softmax=True)
    loss = (1 - args.dice_weight) * loss_ce + args.dice_weight * loss_dice
    return loss, loss_ce, loss_dice


def training_per_epoch(model, trainloader, optimizer, iter_num, ce_loss, dice_loss, multimask_output=True, args=None):
    model.train()
    loss_all = []

    for i_batch, sampled_batch in enumerate(trainloader):
        image_batch, label_batch, low_res_label_batch = sampled_batch['image'],sampled_batch['mask'], sampled_batch['low_res_label']
        image_batch, label_batch, low_res_label_batch = image_batch.to(device, dtype=torch.float32), label_batch.to(device, dtype=torch.long), low_res_label_batch.to(device, dtype=torch.long)
        batch_dict = {'image_batch':label_batch, 'label_batch':label_batch, 'low_res_label_batch':low_res_label_batch}
        outputs = model(image_batch, multimask_output, args.img_size)
        output = outputs[args.output_key]
        loss_label_batch = batch_dict[args.batch_key]
        loss, loss_ce, loss_dice = calc_loss(output, loss_label_batch, ce_loss, dice_loss, args)
        optimizer.zero_grad()
        loss.backward()

        optimizer.step()
        # Update learning rate and increment iteration count
        lr_current = adjust_learning_rate(optimizer, iter_num, args)
        iter_num += 1

        loss_all.append(loss.item())

    return np.mean(loss_all), iter_num, lr_current


def test_per_epoch(model, testloader, ce_loss, dice_loss, multimask_output=True, args=None):
    model.eval()
    loss_per_epoch, dice_per_epoch = [], []
    with torch.no_grad():
        for i_batch, sampled_batch in enumerate(testloader):
            image_batch, label_batch, low_res_label_batch = sampled_batch['image'],sampled_batch['mask'], sampled_batch['low_res_label']
            image_batch, label_batch, low_res_label_batch = image_batch.to(device, dtype=torch.float32), label_batch.to(device, dtype=torch.long), low_res_label_batch.to(device, dtype=torch.long)
            batch_dict = {'image_batch':image_batch, 'label_batch':label_batch, 'low_res_label_batch':low_res_label_batch}
            outputs = model(image_batch, multimask_output, args.img_size)
            output = outputs[args.output_key]
            loss_label_batch = batch_dict[args.batch_key]
            loss, loss_ce, loss_dice = calc_loss(output, loss_label_batch, ce_loss, dice_loss, args)
            loss_per_epoch.append(loss.item())
            dice_per_epoch.append(1-loss_dice.item())
    return np.mean(loss_per_epoch), np.mean(dice_per_epoch)



def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--use_official_metrics', action='store_true', help='Whether to use official metrics for evaluation')
    parser.add_argument('--save_predictions', action='store_true', help='Whether to save predictions for external evaluation')
    
    parser.add_argument('--batch_key', type=str, default='low_res_label_batch', help='Key for accessing label batch')
    parser.add_argument('--output_key', type=str, default='low_res_logits', help='Key for accessing model outputs')

    parser.add_argument('--dice_weight', type=float, default=0.8, help='Weight for dice loss in the loss calculation')
    parser.add_argument('--weights', type=int, nargs='+', default=None,help='List of weights for each class. Provide space-separated values.')

    parser.add_argument('--config', type=str, default=None, help='The config file provided by the trained model')
    # parser.add_argument('--volume_path', type=str, default='/content/samed_codes/test_dataset')
    parser.add_argument('--data_path', type=str, default=r'/SAN/medic/Surgical_LLM_Agent/SAR')    
    # parser.add_argument('--data_path', type=str, default='Endonasal_Slices_Voxel')
    parser.add_argument('--dataset', type=str, default='Synapse', help='Experiment name')
    parser.add_argument('--num_classes', type=int, default=9, help='Number of classes')#exclude backgrounds
    parser.add_argument('--list_dir', type=str, default='./lists/lists_Synapse/', help='list_dir')
    parser.add_argument('--output_dir', type=str, default='weight_results')
    parser.add_argument('--output_file', type=str, default='Endo_best.pt') ############
    parser.add_argument('--img_size', type=int, default=512, help='Input image size of the network')
    parser.add_argument('--input_size', type=int, default=224, help='The input size for training SAM model')
    parser.add_argument('--seed', type=int, default=1234, help='random seed')
    parser.add_argument('--is_savenii', action='store_true', help='Whether to save results during inference')
    parser.add_argument('--deterministic', type=int, default=1, help='whether use deterministic training')
    parser.add_argument('--ckpt', type=str, default='checkpoints/sam_vit_b_01ec64.pth',help='Pretrained checkpoint')
    parser.add_argument('--lora_ckpt', type=str, default='checkpoints/epoch_159.pth', help='The checkpoint from LoRA')
    parser.add_argument('--vit_name', type=str, default='vit_b', help='Select one vit model')
    parser.add_argument('--rank', type=int, default=6, help='Rank for LoRA adaptation')
    parser.add_argument('--module', type=str, default='sam_lora_image_encoder')

    parser.add_argument('--base_lr', type=float, default=0.001, help='segmentation network learning rate')
    parser.add_argument('--batch_size', type=int, default=64, help='batch_size per gpu')
    parser.add_argument('--warmup', type=bool, default=True, help='If activated, warp up the learning from a lower lr to the base_lr')
    parser.add_argument('--warmup_period', type=int, default=250, help='Warp up iterations, only valid whrn warmup is activated')
    parser.add_argument('--AdamW', type=bool, default=True, help='If activated, use AdamW to finetune SAM model')
    parser.add_argument('--max_epochs', type=int, default=100, help='maximum epoch number to train')
    parser.add_argument('--max_iterations', type=int, default=30000, help='maximum epoch number to train')

    if 'ipykernel' in sys.modules:
        args = parser.parse_args([])
    else:
        args = parser.parse_args()

    args.output_dir = 'results'
    args.ckpt = 'sam_vit_b_01ec64.pth'
    args.lora_ckpt = 'results/' + args.output_file
    os.makedirs(args.output_dir, exist_ok = True)

    sam, img_embedding_size = sam_model_registry[args.vit_name](image_size=args.img_size,
                                                                    num_classes=args.num_classes,
                                                                    checkpoint=args.ckpt, pixel_mean=[0, 0, 0],
                                                                    pixel_std=[1, 1, 1])

    # pkg = import_module(args.module)
    net = LoRA_Sam_v0_v2(sam, args.rank).cuda()
    # net.load_lora_parameters(args.lora_ckpt)
    multimask_output = True if args.num_classes > 1 else False

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
        # transforms.ToTensor(),
        ])
# ds = SegmentationDataset(transform_img=transform_img, transform_mask=transform_mask, istrain=True)

    train_dataset = SegmentationDataset(root_dir=args.data_path, subset='Train', low_res=128, transform_img=transform_img, transform_mask=transform_mask, istrain=True)
    test_dataset = SegmentationDataset(root_dir=args.data_path, subset='Validate', low_res=128, transform_img=transform_img, transform_mask=transform_mask)
    trainloader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=2)
    testloader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False, num_workers=2)
    print('Training on:', device, 'train sample size:', len(train_dataset), 'test sample size:', len(test_dataset), 'batch:', args.batch_size)

    ce_loss = CrossEntropyLoss()
    dice_loss = DiceLoss(args.num_classes + 1)
    b_lr = args.base_lr / args.warmup_period
    optimizer = optim.AdamW(filter(lambda p: p.requires_grad, net.parameters()), lr=b_lr, betas=(0.9, 0.999), weight_decay=0.1)
    iter_num = 0

    # # test if there is low_res_label
    # for i_batch, sampled_batch in enumerate(trainloader):
    #   print(f"Sampled batch keys: {sampled_batch.keys()}") 

    best_epoch, best_loss = 0.0, np.inf
    for epoch in range(args.max_epochs):
        loss_training, iter_num, lr_current = training_per_epoch(net, trainloader, optimizer, iter_num, ce_loss, dice_loss, multimask_output=multimask_output, args=args)
        loss_testing, dice = test_per_epoch(net, testloader, ce_loss, dice_loss,multimask_output=True, args=args)

        if loss_testing < best_loss:
            best_loss = loss_testing
            best_epoch = epoch
            net.save_lora_parameters(os.path.join(args.output_dir, args.output_file))

        print('--- Epoch {}/{}: Training loss = {:.4f}, Testing: [loss = {:.4f}, dice = {:.4f}], Best loss = {:.4f}, Best epoch = {}, lr = {:.6f}'.\
    format(epoch, args.max_epochs, loss_training, loss_testing, dice, best_loss, best_epoch, lr_current))

    assert args.lora_ckpt is not None
    net.load_lora_parameters(args.lora_ckpt)
    testloader = DataLoader(test_dataset, batch_size=20, shuffle=False, num_workers=2)
    

    test_loss, overall_dice, dices_per_class, official_metrics = inference_per_epoch(
        net, testloader, ce_loss, dice_loss, multimask_output=True, args=args, 
        save_predictions=args.save_predictions
    )
    
    dices_per_class_list = np.array(list(dices_per_class.values()))
    print('Class Wise Dice:', dices_per_class)
    print('Overall Dice:', np.mean(dices_per_class_list))
    
    if official_metrics:
        print('\n--- Official Evaluation Metrics ---')
        print(f'mIoU: {official_metrics["mIoU"]:.4f}')
        print(f'mNSD: {official_metrics["mNSD"]:.4f}')
        print(f'Final Score: {official_metrics["Final_Score"]:.4f}')

if __name__ == '__main__':
    seed_everything()
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(device)
    main()