import os
import cv2
import sys
import argparse
import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset
import torch.nn.functional as F
from segment_anything import sam_model_registry
from pathlib import Path
import matplotlib.pyplot as plt
from tqdm import tqdm
import monai
import warnings
from torchvision.transforms import InterpolationMode
import torchvision.transforms as transforms
from PIL import Image
from glob import glob
import shutil
from collections import defaultdict
import matplotlib.colors as mcolors
from matplotlib.patches import Patch
import seaborn as sns

# Import the LoRA implementation
class _LoRA_qkv_v0_v2(torch.nn.Module):
    def __init__(
            self,
            qkv: torch.nn.Module,
            linear_a_q: torch.nn.Module,
            linear_b_q: torch.nn.Module,
            linear_a_v: torch.nn.Module,
            linear_b_v: torch.nn.Module,
            conv_se_q: torch.nn.Module,
            conv_se_v: torch.nn.Module,
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

        new_q = self.linear_b_q(torch.mul(a_q_out, torch.sigmoid(a_q_out_temp)))
        new_v = self.linear_b_v(torch.mul(a_v_out, torch.sigmoid(a_v_out_temp)))

        qkv[:, :, :, : self.dim] += new_q
        qkv[:, :, :, -self.dim:] += new_v
        return qkv

class LoRA_Sam_v0_v2(torch.nn.Module):
    def __init__(self, sam_model, r: int, lora_layer=None):
        super(LoRA_Sam_v0_v2, self).__init__()

        assert r > 0
        if lora_layer:
            self.lora_layer = lora_layer
        else:
            self.lora_layer = list(
                range(len(sam_model.image_encoder.blocks)))
        
        self.w_As = []
        self.w_Bs = []

        for param in sam_model.image_encoder.parameters():
            param.requires_grad = False

        for t_layer_i, blk in enumerate(sam_model.image_encoder.blocks):
            if t_layer_i not in self.lora_layer:
                continue
            w_qkv_linear = blk.attn.qkv
            self.dim = w_qkv_linear.in_features
            w_a_linear_q = torch.nn.Linear(self.dim, r, bias=False)
            w_b_linear_q = torch.nn.Linear(r, self.dim, bias=False)
            w_a_linear_v = torch.nn.Linear(self.dim, r, bias=False)
            w_b_linear_v = torch.nn.Linear(r, self.dim, bias=False)

            conv_se_q = torch.nn.Conv2d(r, r, kernel_size=1,
                                    stride=1, padding=0, bias=False)
            conv_se_v = torch.nn.Conv2d(r, r, kernel_size=1,
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
        num_layer = len(self.w_As)
        a_tensors = {f"w_a_{i:03d}": self.w_As[i].weight for i in range(num_layer)}
        num_layer = len(self.w_Bs)
        b_tensors = {f"w_b_{i:03d}": self.w_Bs[i].weight for i in range(num_layer)}
        prompt_encoder_tensors = {}
        mask_decoder_tensors = {}

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
        state_dict = torch.load(filename, weights_only=False)

        for i, w_A_linear in enumerate(self.w_As):
            saved_key = f"w_a_{i:03d}"
            saved_tensor = state_dict[saved_key]
            w_A_linear.weight = torch.nn.Parameter(saved_tensor)

        for i, w_B_linear in enumerate(self.w_Bs):
            saved_key = f"w_b_{i:03d}"
            saved_tensor = state_dict[saved_key]
            w_B_linear.weight = torch.nn.Parameter(saved_tensor)

        sam_dict = self.sam.state_dict()
        sam_keys = sam_dict.keys()

        prompt_encoder_keys = [k for k in sam_keys if 'prompt_encoder' in k]
        prompt_encoder_values = [state_dict[k] for k in prompt_encoder_keys]
        prompt_encoder_new_state_dict = {k: v for k, v in zip(prompt_encoder_keys, prompt_encoder_values)}
        sam_dict.update(prompt_encoder_new_state_dict)

        mask_decoder_keys = [k for k in sam_keys if 'mask_decoder' in k]
        mask_decoder_values = [state_dict[k] for k in mask_decoder_keys]
        mask_decoder_new_state_dict = {k: v for k, v in zip(mask_decoder_keys, mask_decoder_values)}
        sam_dict.update(mask_decoder_new_state_dict)
        self.sam.load_state_dict(sam_dict)

    def reset_parameters(self) -> None:
        import math
        for w_A in self.w_As:
            torch.nn.init.kaiming_uniform_(w_A.weight, a=math.sqrt(5))
        for w_B in self.w_Bs:
            torch.nn.init.zeros_(w_B.weight)

    def forward(self, batched_input, multimask_output, image_size):
        return self.sam(batched_input, multimask_output, image_size)

class VideoSegmentationDataset(Dataset):
    def __init__(self, root_dir, video_dir, transform_img=None, transform_mask=None, low_res=None):
        """
        Dataset loader for a specific video directory
        
        Args:
            root_dir: Dataset root directory
            video_dir: Specific video directory name
            transform_img: Image transformations
            transform_mask: Mask transformations
            low_res: Low resolution label size
        """
        self.root_dir = root_dir
        self.video_dir = video_dir
        self.video_path = os.path.join(root_dir, video_dir)
        
        self.rgb_dir = os.path.join(self.video_path, 'images')
        self.seg_dir = os.path.join(self.video_path, 'segmentation')
        
        # Get all image and segmentation pairs
        self.img_paths = []
        self.mask_paths = []
        
        # Get segmentation files
        seg_files = sorted(glob(os.path.join(self.seg_dir, '*.png')))
        
        for seg_file in seg_files:
            frame_num = os.path.basename(seg_file)
            rgb_file = os.path.join(self.rgb_dir, frame_num)
            
            if os.path.exists(rgb_file):
                self.img_paths.append(rgb_file)
                self.mask_paths.append(seg_file)
        
        self.transform_img = transform_img
        self.transform_mask = transform_mask
        self.low_res = low_res
        
    def __len__(self):
        return len(self.img_paths)
    
    def __getitem__(self, idx):
        img_path = self.img_paths[idx]
        mask_path = self.mask_paths[idx]
        
        # Open image and mask
        image = Image.open(img_path).convert('RGB')
        mask = Image.open(mask_path).convert('L')
        mask = np.array(mask)
        mask = Image.fromarray(mask)
        
        # Apply transformations
        if self.transform_img:
            image = self.transform_img(image)
            
        if self.transform_mask:
            mask = self.transform_mask(mask)
            
        # Convert to tensor
        mask = torch.from_numpy(np.array(mask)).long()
        
        # Create sample dictionary
        sample = {'image': image, 'mask': mask, 'path': img_path}
        
        # Add low resolution label if needed
        if self.low_res:
            from scipy.ndimage import zoom
            low_res_label = zoom(mask.numpy(), (self.low_res/mask.shape[0], self.low_res/mask.shape[1]), order=0)
            low_res_label = torch.from_numpy(low_res_label).long()
            sample['low_res_label'] = low_res_label
            
        return sample

def imread_one_hot(filepath, n_classes):
    # Reads a segmentation mask and returns it in one-hot format
    img = cv2.imread(str(filepath))
    if img is None:
        raise FileNotFoundError(filepath)
    if len(img.shape) == 3:
        img = img[..., 0]
    img = torch.from_numpy(img).unsqueeze(0).unsqueeze(0).requires_grad_(False)
    return monai.networks.utils.one_hot(img, n_classes, dim=1)

def save_one_hot(root_dir, oh):
    # Saves a one-hot tensor as separate images
    for i, c in enumerate(oh):
        cv2.imwrite(f'{root_dir}/{i}.png', c.numpy().astype(np.uint8)*255)

def get_val_func(metric, n_classes=9, fix_nans=False):
    def f(dir_pred, dir_ref):
        seg_ref_paths = sorted(list(dir_ref.iterdir()))
        dir_pred = Path(dir_pred)
        
        acc = []
        with torch.no_grad():
            for seg_ref_p in seg_ref_paths:
                try:
                    ref = imread_one_hot(seg_ref_p, n_classes=n_classes+1)
                except FileNotFoundError:
                    raise
                try:
                    pred = imread_one_hot(dir_pred/seg_ref_p.name, n_classes=n_classes+1)
                except FileNotFoundError as e:
                    acc.append([0]*n_classes)
                    continue
                
                if fix_nans:
                    with warnings.catch_warnings():
                        warnings.simplefilter("ignore")
                        err = metric(pred, ref)
                    r_m, p_m = ref.mean(axis=(2,3)), pred.mean(axis=(2,3))
                    mask = ((r_m == 0) * (p_m == 0))[:,1:]
                    err[mask==True] = 1
                    err.nan_to_num_()
                else:
                    err = metric(pred, ref)
                acc.append(err.detach().reshape(-1).tolist())
        return np.array(acc)
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

def evaluate_video(model, video_loader, device, args, video_name):
    """
    Evaluate model on a single video and compute metrics
    """
    model.eval()
    num_classes = args.num_classes + 1
    confusion_matrix = np.zeros((num_classes, num_classes), dtype=np.uint32)
    
    # Create directories for evaluation and results
    output_dir = Path(args.output_dir)
    
    # Directory for metrics calculation
    temp_pred_dir = output_dir / "temp_pred"
    temp_ref_dir = output_dir / "temp_ref"
    (temp_pred_dir / "segmentation").mkdir(parents=True, exist_ok=True)
    (temp_ref_dir / "segmentation").mkdir(parents=True, exist_ok=True)
    
    # Directory for all predictions
    all_predictions_dir = output_dir / "all_predictions" / video_name
    all_predictions_dir.mkdir(parents=True, exist_ok=True)
    all_predictions_hires_dir = output_dir / "all_predictions_hires" / video_name
    all_predictions_hires_dir.mkdir(parents=True, exist_ok=True)
    
    # Directory for all original images
    all_originals_dir = output_dir / "all_originals" / video_name
    all_originals_dir.mkdir(parents=True, exist_ok=True)
    all_originals_hires_dir = output_dir / "all_originals_hires" / video_name
    all_originals_hires_dir.mkdir(parents=True, exist_ok=True)
    
    # Directory for all ground truth masks
    all_gt_dir = output_dir / "all_ground_truth" / video_name
    all_gt_dir.mkdir(parents=True, exist_ok=True)
    all_gt_hires_dir = output_dir / "all_ground_truth_hires" / video_name
    all_gt_hires_dir.mkdir(parents=True, exist_ok=True)
    
    # Track best performing image
    best_score = -1
    best_sample = None
    best_pred = None
    best_idx = None
    
    # Define class names and colors
    class_names = [
        'Background', 
        'Tool Clasper', 
        'Tool Wrist', 
        'Tool Shaft', 
        'Suturing Needle', 
        'Thread', 
        'Suction Tool', 
        'Needle Holder', 
        'Clamps', 
        'Catheter'
    ]
    custom_colors = [
        '#000000',  # Background (black)
        '#00FFFF',  # Tool Clasper (cyan)
        '#00FF00',  # Tool Wrist (bright green)
        '#0000FF',  # Tool Shaft (blue)
        '#FFFF00',  # Suturing Needle (yellow)
        '#DAA520',  # Thread (goldenrod)
        '#800080',  # Suction Tool (purple)
        '#FFA500',  # Needle Holder (orange)
        '#FF0000',  # Clamps (red)
        '#808080'   # Catheter (gray)
    ]

    # Create custom color map
    custom_cmap = mcolors.ListedColormap(custom_colors)
    
    # Store predictions and references for metric calculation
    with torch.no_grad():
        for i, sample in enumerate(tqdm(video_loader, desc=f"Processing frames for {video_name}")):
            image = sample['image'].to(device)
            mask = sample['mask'].to(device)
            low_res_label = sample.get('low_res_label', None)
            if low_res_label is not None:
                low_res_label = low_res_label.to(device)
            
            # Forward pass
            outputs = model(image, args.multimask_output, args.img_size)
            logits = outputs['masks']
            prob = F.softmax(logits, dim=1)
            pred = torch.argmax(prob, dim=1)
            
            # Update confusion matrix
            confusion_matrix += calculate_confusion_matrix_from_arrays(
                pred.cpu().numpy(), mask.cpu().numpy(), num_classes)
            
            
            # Save predictions and references for metric calculation and storage
            for j in range(pred.shape[0]):
                img_idx = i * image.shape[0] + j
                
                # Get cpu versions of tensors
                pred_cpu = pred[j].cpu().numpy().astype(np.uint8)
                mask_cpu = mask[j].cpu().numpy().astype(np.uint8)
                
                # Save prediction and reference for metric calculation
                cv2.imwrite(str(temp_pred_dir / "segmentation" / f"{img_idx:04d}.png"), pred_cpu)
                cv2.imwrite(str(temp_ref_dir / "segmentation" / f"{img_idx:04d}.png"), mask_cpu)
                
                # Save all predictions, originals, and ground truth
                frame_base_name = f"{img_idx:04d}"
                        
                # Save prediction with color visualization
                pred_vis = custom_cmap(pred_cpu)
                plt.imsave(str(all_predictions_dir / f"{frame_base_name}_pred.png"), pred_vis)
                pred_img = Image.fromarray((pred_vis * 255).astype(np.uint8))
                pred_img_resized = pred_img.resize((1920, 1080), Image.NEAREST)
                pred_img_resized.save(str(all_predictions_hires_dir / f"{frame_base_name}_pred_hires.png"))
                
                # Save raw prediction
                cv2.imwrite(str(all_predictions_dir / f"{frame_base_name}_pred_raw.png"), pred_cpu)
                pred_raw_img = Image.fromarray((pred_cpu * 255).astype(np.uint8))
                pred_raw_img_resized = pred_raw_img.resize((1920, 1080), Image.NEAREST)
                pred_raw_img_resized.save(str(all_predictions_hires_dir / f"{frame_base_name}_pred_raw_hires.png"))
                
                # Save ground truth with color visualization
                gt_vis = custom_cmap(mask_cpu)
                plt.imsave(str(all_gt_dir / f"{frame_base_name}_gt.png"), gt_vis)
                gt_img = Image.fromarray((gt_vis * 255).astype(np.uint8))
                gt_img_resized = gt_img.resize((1920, 1080), Image.NEAREST)
                gt_img_resized.save(str(all_gt_hires_dir / f"{frame_base_name}_gt_hires.png"))
                
                # Save raw ground truth
                cv2.imwrite(str(all_gt_dir / f"{frame_base_name}_gt_raw.png"), mask_cpu)
                gt_raw_img = Image.fromarray((mask_cpu * 255).astype(np.uint8))
                gt_raw_img_resized = gt_raw_img.resize((1920, 1080), Image.NEAREST)
                gt_raw_img_resized.save(str(all_gt_hires_dir / f"{frame_base_name}_gt_raw_hires.png"))
                
                # Save original image
                img_np = image[j].cpu()
                # Denormalize
                mean = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
                std = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)
                img_np = img_np * std + mean
                img_np = torch.clamp(img_np, 0, 1)
                img_np = img_np.permute(1, 2, 0).numpy()
                plt.imsave(str(all_originals_dir / f"{frame_base_name}_original.png"), img_np)
                orig_img = Image.fromarray((img_np * 255).astype(np.uint8))
                orig_img_resized = orig_img.resize((1920, 1080), Image.LANCZOS)
                orig_img_resized.save(str(all_originals_hires_dir / f"{frame_base_name}_original_hires.png"))
                
                # Calculate individual Dice score for this image to find best one
                current_cm = calculate_confusion_matrix_from_arrays(pred_cpu, mask_cpu, num_classes)
                current_cm = current_cm[1:, 1:]  # exclude background
                current_dice = np.mean(calculate_dice(current_cm))
                
                if current_dice > best_score:
                    best_score = current_dice
                    best_sample = {
                        'image': image[j].cpu(),
                        'mask': mask[j].cpu(),
                        'path': sample['path'][j] if isinstance(sample['path'], list) else sample['path']
                    }
                    best_pred = pred[j].cpu()
                    best_idx = img_idx
    
    # Copy the best performing frame to a separate directory
    best_frames_dir = output_dir / "best_frames" / video_name
    best_frames_dir.mkdir(parents=True, exist_ok=True)
    
    if best_idx is not None:
        # Copy best prediction
        best_pred_path = all_predictions_dir / f"{best_idx:04d}_pred.png"
        if best_pred_path.exists():
            shutil.copy(best_pred_path, best_frames_dir / "best_pred.png")
        
        # Copy best raw prediction
        best_pred_raw_path = all_predictions_dir / f"{best_idx:04d}_pred_raw.png"
        if best_pred_raw_path.exists():
            shutil.copy(best_pred_raw_path, best_frames_dir / "best_pred_raw.png")
        
        # Copy best ground truth
        best_gt_path = all_gt_dir / f"{best_idx:04d}_gt.png"
        if best_gt_path.exists():
            shutil.copy(best_gt_path, best_frames_dir / "best_gt.png")
        
        # Copy best ground truth raw
        best_gt_raw_path = all_gt_dir / f"{best_idx:04d}_gt_raw.png"
        if best_gt_raw_path.exists():
            shutil.copy(best_gt_raw_path, best_frames_dir / "best_gt_raw.png")
        
        # Copy best original
        best_original_path = all_originals_dir / f"{best_idx:04d}_original.png"
        if best_original_path.exists():
            shutil.copy(best_original_path, best_frames_dir / "best_original.png")
            
        # Copy high-resolution versions
        best_pred_hires_path = all_predictions_hires_dir / f"{best_idx:04d}_pred_hires.png"
        if best_pred_hires_path.exists():
            shutil.copy(best_pred_hires_path, best_frames_dir / "best_pred_hires.png")
        
        best_pred_raw_hires_path = all_predictions_hires_dir / f"{best_idx:04d}_pred_raw_hires.png"
        if best_pred_raw_hires_path.exists():
            shutil.copy(best_pred_raw_hires_path, best_frames_dir / "best_pred_raw_hires.png")
        
        best_gt_hires_path = all_gt_hires_dir / f"{best_idx:04d}_gt_hires.png"
        if best_gt_hires_path.exists():
            shutil.copy(best_gt_hires_path, best_frames_dir / "best_gt_hires.png")
        
        best_gt_raw_hires_path = all_gt_hires_dir / f"{best_idx:04d}_gt_raw_hires.png"
        if best_gt_raw_hires_path.exists():
            shutil.copy(best_gt_raw_hires_path, best_frames_dir / "best_gt_raw_hires.png")
        
        best_original_hires_path = all_originals_hires_dir / f"{best_idx:04d}_original_hires.png"
        if best_original_hires_path.exists():
            shutil.copy(best_original_hires_path, best_frames_dir / "best_original_hires.png")
        
        # Create a combined visualization
        if best_sample is not None and best_pred is not None:
            # Create the combined visualization directly in the best_frames directory
            fig, axes = plt.subplots(1, 3, figsize=(18, 6))
            
            # Denormalize image for visualization
            image_np = best_sample['image']
            if len(image_np.shape) == 3 and image_np.shape[0] == 3:
                mean = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
                std = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)
                image_np = image_np * std + mean
                image_np = torch.clamp(image_np, 0, 1)
            image_np = image_np.permute(1, 2, 0).numpy()
            
            # Plot original image
            axes[0].imshow(image_np)
            axes[0].set_title('Original Image')
            axes[0].axis('off')
            
            # Plot ground truth mask
            axes[1].imshow(best_sample['mask'].numpy(), cmap=custom_cmap, vmin=0, vmax=9)
            axes[1].set_title('Ground Truth Mask')
            axes[1].axis('off')
            
            # Plot predicted mask
            axes[2].imshow(best_pred.numpy(), cmap=custom_cmap, vmin=0, vmax=9)
            axes[2].set_title('Predicted Mask')
            axes[2].axis('off')
            
            legend_patches = [Patch(color=custom_colors[i], label=class_names[i]) for i in range(len(class_names))]
            fig.legend(handles=legend_patches, loc='lower center', ncol=5, bbox_to_anchor=(0.5, -0.1))
            
            # Add metrics to title
            fig.suptitle(f"Best frame (Dice: {best_score:.4f})")
            
            plt.tight_layout()
            plt.savefig(str(best_frames_dir / "combined.png"), dpi=200, bbox_inches='tight')
            plt.close()
            
            # Save frame index information
            with open(str(best_frames_dir / "info.txt"), "w") as f:
                f.write(f"Best frame index: {best_idx}\n")
                f.write(f"Dice score: {best_score:.4f}\n")
                if 'path' in best_sample:
                    f.write(f"Original path: {best_sample['path']}\n")
                    
        # Create a combined visualization for high-resolution images
        if best_pred_hires_path.exists() and best_gt_hires_path.exists() and best_original_hires_path.exists():
            try:
                # Load high-resolution images
                pred_hires_img = Image.open(best_pred_hires_path)
                gt_hires_img = Image.open(best_gt_hires_path)
                original_hires_img = Image.open(best_original_hires_path)
                
                # Convert to numpy arrays for matplotlib
                pred_hires = np.array(pred_hires_img)
                gt_hires = np.array(gt_hires_img)
                original_hires = np.array(original_hires_img)
                
                # Create figure for high-resolution combined visualization
                fig, axes = plt.subplots(1, 3, figsize=(19.2, 10.8))
                
                # Plot original high-res image
                axes[0].imshow(original_hires)
                axes[0].set_title('Original Image', fontsize=14)
                axes[0].axis('off')
                
                # Plot ground truth high-res mask
                axes[1].imshow(gt_hires)
                axes[1].set_title('Ground Truth Mask', fontsize=14)
                axes[1].axis('off')
                
                # Plot predicted high-res mask
                axes[2].imshow(pred_hires)
                axes[2].set_title('Predicted Mask', fontsize=14)
                axes[2].axis('off')
                
                # Add legend
                legend_patches = [Patch(color=custom_colors[i], label=class_names[i]) for i in range(len(class_names))]
                fig.legend(handles=legend_patches, loc='lower center', ncol=5, bbox_to_anchor=(0.5, -0.1), fontsize=12)
                
                # Add metrics to title
                fig.suptitle(f"Best frame (Dice: {best_score:.4f})", fontsize=18)
                
                # Save the high-resolution combined visualization
                plt.tight_layout(rect=[0, 0.05, 1, 0.95])
                plt.savefig(str(best_frames_dir / "combined_hires.png"), dpi=200, bbox_inches='tight')
                plt.close()
            except Exception as e:
                print(f"Error creating high-resolution combined visualization: {e}")
    
    # Calculate official metrics - now storing per-class metrics
    try:
        # Calculate mIoU (per class)
        miou_scores = mIoU(temp_pred_dir, temp_ref_dir, n_classes=args.num_classes)
        miou_per_class = np.mean(miou_scores, axis=0)
        mean_miou = np.mean(miou_per_class)
        
        # Calculate mNSD (per class)
        mnsd_scores = mNSD(temp_pred_dir, temp_ref_dir, n_classes=args.num_classes)
        mnsd_per_class = np.mean(mnsd_scores, axis=0)
        mean_mnsd = np.mean(mnsd_per_class)
        
        final_score = np.sqrt(mean_miou * mean_mnsd)
        
        # Calculate per-class final scores
        final_scores_per_class = np.sqrt(miou_per_class * mnsd_per_class)
        
        metrics = {
            'mIoU': mean_miou,
            'mNSD': mean_mnsd,
            'Final_Score': final_score,
            'mIoU_per_class': miou_per_class.tolist(),
            'mNSD_per_class': mnsd_per_class.tolist(),
            'Final_Score_per_class': final_scores_per_class.tolist()
        }
        
    except Exception as e:
        print(f"Error computing official metrics: {e}")
        metrics = {
            'mIoU': 0.0,
            'mNSD': 0.0,
            'Final_Score': 0.0,
            'mIoU_per_class': [0.0] * args.num_classes,
            'mNSD_per_class': [0.0] * args.num_classes,
            'Final_Score_per_class': [0.0] * args.num_classes
        }
    
    # Calculate dice from confusion matrix
    confusion_matrix_no_bg = confusion_matrix[1:, 1:]  # exclude background
    dices_per_class = calculate_dice(confusion_matrix_no_bg)
    mean_dice = np.mean(dices_per_class)
    
    # Save metrics for this video
    metrics_dir = output_dir / "metrics"
    metrics_dir.mkdir(exist_ok=True)
    metrics_file = metrics_dir / f"{video_name}_metrics.json"
    
    import json
    with open(metrics_file, 'w') as f:
        json.dump({
            'mIoU': float(metrics['mIoU']),
            'mNSD': float(metrics['mNSD']),
            'Final_Score': float(metrics['Final_Score']),
            'Dice': float(mean_dice),
            'Dice_per_class': [float(x) for x in dices_per_class],
            'mIoU_per_class': metrics['mIoU_per_class'],
            'mNSD_per_class': metrics['mNSD_per_class'],
            'Final_Score_per_class': metrics['Final_Score_per_class'],
            'Best_frame_idx': int(best_idx) if best_idx is not None else None,
            'Best_frame_dice': float(best_score)
        }, f, indent=4)
    
    # Create per-class metrics visualization
    # Generate bar charts for per-class metrics
    per_class_dir = output_dir / "per_class_metrics" / video_name
    per_class_dir.mkdir(parents=True, exist_ok=True)
    
    # Class names without background
    class_names_no_bg = class_names[1:]
    
    # Plot per-class metrics
    fig, axes = plt.subplots(2, 2, figsize=(16, 14))
    
    # Plot mIoU per class
    axes[0, 0].bar(class_names_no_bg, metrics['mIoU_per_class'], color='skyblue')
    axes[0, 0].set_title(f'mIoU per Class - {video_name}', fontsize=14)
    axes[0, 0].set_ylabel('mIoU')
    axes[0, 0].set_ylim(0, 1.0)
    for i, v in enumerate(metrics['mIoU_per_class']):
        if v > 0.05:  
            text_y = v/2 
            axes[0, 0].text(i, text_y, f'{v:.3f}', ha='center', va='center', fontsize=9, color='black', fontweight='bold')
        # axes[0, 0].text(i, v + 0.02, f'{v:.3f}', ha='center', fontsize=10)
    axes[0, 0].axhline(y=metrics['mIoU'], color='r', linestyle='-', label=f'Average: {metrics["mIoU"]:.3f}')
    axes[0, 0].legend()
    plt.setp(axes[0, 0].xaxis.get_majorticklabels(), rotation=45, ha='right')
    
    # Plot mNSD per class
    axes[0, 1].bar(class_names_no_bg, metrics['mNSD_per_class'], color='lightgreen')
    axes[0, 1].set_title(f'mNSD per Class - {video_name}', fontsize=14)
    axes[0, 1].set_ylabel('mNSD')
    axes[0, 1].set_ylim(0, 1.0)
    for i, v in enumerate(metrics['mNSD_per_class']):
        if v > 0.05:  # 
            text_y = v/2  
            axes[0, 1].text(i, text_y, f'{v:.3f}', ha='center', va='center', fontsize=9, color='black', fontweight='bold')
        # axes[0, 1].text(i, v + 0.02, f'{v:.3f}', ha='center', fontsize=10)
    axes[0, 1].axhline(y=metrics['mNSD'], color='r', linestyle='-', label=f'Average: {metrics["mNSD"]:.3f}')
    axes[0, 1].legend()
    plt.setp(axes[0, 1].xaxis.get_majorticklabels(), rotation=45, ha='right')
    
    # Plot final score per class
    axes[1, 0].bar(class_names_no_bg, metrics['Final_Score_per_class'], color='coral')
    axes[1, 0].set_title(f'Final Score per Class - {video_name}', fontsize=14)
    axes[1, 0].set_ylabel('Final Score')
    axes[1, 0].set_ylim(0, 1.0)
    for i, v in enumerate(metrics['Final_Score_per_class']):
        if v > 0.05:  # 
            text_y = v/2  
            axes[1, 0].text(i, text_y, f'{v:.3f}', ha='center', va='center', fontsize=9, color='black', fontweight='bold')
        # axes[1, 0].text(i, v + 0.02, f'{v:.3f}', ha='center', fontsize=10)
    axes[1, 0].axhline(y=metrics['Final_Score'], color='r', linestyle='-', label=f'Average: {metrics["Final_Score"]:.3f}')
    axes[1, 0].legend()
    plt.setp(axes[1, 0].xaxis.get_majorticklabels(), rotation=45, ha='right')
    
    # Plot dice per class
    axes[1, 1].bar(class_names_no_bg, dices_per_class, color='plum')
    axes[1, 1].set_title(f'Dice Score per Class - {video_name}', fontsize=14)
    axes[1, 1].set_ylabel('Dice Score')
    axes[1, 1].set_ylim(0, 1.0)
    for i, v in enumerate(dices_per_class):
        if v > 0.05:  # 
            text_y = v/2  
            axes[1, 1].text(i, text_y, f'{v:.3f}', ha='center', va='center', fontsize=9, color='black', fontweight='bold')
        # axes[1, 1].text(i, v + 0.02, f'{v:.3f}', ha='center', fontsize=10)
    axes[1, 1].axhline(y=mean_dice, color='r', linestyle='-', label=f'Average: {mean_dice:.3f}')
    axes[1, 1].legend()
    plt.setp(axes[1, 1].xaxis.get_majorticklabels(), rotation=45, ha='right')
    
    plt.tight_layout()
    plt.savefig(str(per_class_dir / "per_class_metrics.png"), dpi=200, bbox_inches='tight')
    plt.close()
    
    # Clean up temporary directories for metric calculation
    if not args.keep_temp_files:
        shutil.rmtree(temp_pred_dir)
        shutil.rmtree(temp_ref_dir)
        
    dices_per_class_list = [float(x) for x in dices_per_class]
    
    # Return all metrics
    return {
        'metrics': metrics,
        'dice': mean_dice,
        'dice_per_class': dices_per_class_list,
        'best_score': best_score,
        'best_sample': best_sample,
        'best_pred': best_pred,
        'best_idx': best_idx
    }

def main():
    parser = argparse.ArgumentParser(description='Inference on test set videos')
    parser.add_argument('--data_path', type=str, default='/SAN/medic/Surgical_LLM_Agent/SAR', 
                        help='Path to dataset directory')
    parser.add_argument('--subset', type=str, default='Test_ori', help='Dataset subset to use (Test)')
    parser.add_argument('--ckpt', type=str, default='sam_vit_b_01ec64.pth',
                        help='Path to the SAM checkpoint')
    parser.add_argument('--lora_ckpt', type=str, default='results/Endo_best-e45.pt',
                        help='Path to the fine-tuned LoRA weights')
    parser.add_argument('--output_dir', type=str, default='inference_results_epoch45_ver2',
                        help='Directory to save results')
    parser.add_argument('--batch_size', type=int, default=64, help='Batch size for inference')
    parser.add_argument('--img_size', type=int, default=512, help='Input image size')
    parser.add_argument('--num_classes', type=int, default=9, help='Number of classes (excluding background)')
    parser.add_argument('--vit_name', type=str, default='vit_b', help='ViT model size')
    parser.add_argument('--rank', type=int, default=6, help='Rank for LoRA adaptation')
    parser.add_argument('--keep_temp_files', action='store_true', 
                        help='Keep temporary files used for metric calculation')
    parser.add_argument('--save_all_predictions', action='store_true',
                        help='Save all predictions, not just the best one')
    
    args = parser.parse_args()
    args.multimask_output = True if args.num_classes > 1 else False
    
    # Make output directory
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Set device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Load model
    sam, _ = sam_model_registry[args.vit_name](
        image_size=args.img_size,
        num_classes=args.num_classes,
        checkpoint=args.ckpt,
        pixel_mean=[0, 0, 0],
        pixel_std=[1, 1, 1]
    )
    
    # Initialize LoRA model
    model = LoRA_Sam_v0_v2(sam, args.rank).to(device)
    
    # Load trained weights
    model.load_lora_parameters(args.lora_ckpt)
    print(f"Loaded LoRA weights from {args.lora_ckpt}")
    
    # Define transforms
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
    
    # Get all video directories in the test set
    test_dir = os.path.join(args.data_path, args.subset)
    video_dirs = []
    
    for d in os.listdir(test_dir):
        video_path = os.path.join(test_dir, d)
        if os.path.isdir(video_path) and \
           os.path.exists(os.path.join(video_path, 'images')) and \
           os.path.exists(os.path.join(video_path, 'segmentation')):
            video_dirs.append(d)
    
    print(f"Found {len(video_dirs)} videos in test set")
    
    # Evaluate each video and store results
    all_results = {}
    video_metrics = {}
    
    # Define class names
    class_names = [
        'Background', 
        'Tool Clasper', 
        'Tool Wrist', 
        'Tool Shaft', 
        'Suturing Needle', 
        'Thread', 
        'Suction Tool', 
        'Needle Holder', 
        'Clamps', 
        'Catheter'
    ]
    
# Create necessary directories
    output_dir = Path(args.output_dir)
    (output_dir / "all_predictions").mkdir(parents=True, exist_ok=True)
    (output_dir / "all_originals").mkdir(parents=True, exist_ok=True)
    (output_dir / "all_ground_truth").mkdir(parents=True, exist_ok=True)
    (output_dir / "best_frames").mkdir(parents=True, exist_ok=True)
    (output_dir / "metrics").mkdir(parents=True, exist_ok=True)
    (output_dir / "per_class_metrics").mkdir(parents=True, exist_ok=True)
    
    # To store per-class metrics across all videos
    all_miou_per_class = []
    all_mnsd_per_class = []
    all_final_score_per_class = []
    all_dice_per_class = []
    
    for video_dir in tqdm(video_dirs, desc="Processing videos"):
        print(f"\n--- Evaluating video: {video_dir} ---")
        
        # Create dataset for this video
        video_dataset = VideoSegmentationDataset(
            root_dir=test_dir,
            video_dir=video_dir,
            transform_img=transform_img,
            transform_mask=transform_mask,
            low_res=128
        )
        
        # Create dataloader
        video_loader = DataLoader(
            video_dataset,
            batch_size=args.batch_size,
            shuffle=False,
            num_workers=4
        )
        
        print(f"Video {video_dir} has {len(video_dataset)} frames")
        
        # Skip if no frames
        if len(video_dataset) == 0:
            print(f"Skipping video {video_dir} - no valid frames")
            continue
        
        # Evaluate model on this video
        results = evaluate_video(model, video_loader, device, args, video_dir)
        
        # Store results
        all_results[video_dir] = results
        
        # Store metrics
        metrics = results['metrics']
        video_metrics[video_dir] = {
            'mIoU': metrics['mIoU'],
            'mNSD': metrics['mNSD'],
            'Final_Score': metrics['Final_Score'],
            'Dice': results['dice'],
            'mIoU_per_class': metrics['mIoU_per_class'],
            'mNSD_per_class': metrics['mNSD_per_class'],
            'Final_Score_per_class': metrics['Final_Score_per_class'],
            'Dice_per_class': results['dice_per_class'],
            'num_frames': len(video_dataset),
            'best_frame_idx': results.get('best_idx', None),
            'best_frame_dice': results['best_score']
        }
        
        # Store per-class metrics for global analysis
        all_miou_per_class.append(metrics['mIoU_per_class'])
        all_mnsd_per_class.append(metrics['mNSD_per_class'])
        all_final_score_per_class.append(metrics['Final_Score_per_class'])
        all_dice_per_class.append(results['dice_per_class'])
        
        # Print metrics
        print(f"Video: {video_dir}")
        print(f"  mIoU: {metrics['mIoU']:.4f}")
        print(f"  mNSD: {metrics['mNSD']:.4f}")
        print(f"  Final Score: {metrics['Final_Score']:.4f}")
        print(f"  Dice: {results['dice']:.4f}")
        print(f"  Best frame index: {results.get('best_idx', 'N/A')}")
        print(f"  Best frame Dice: {results['best_score']:.4f}")
    
    # Calculate overall metrics (weighted by number of frames)
    total_frames = sum(vm['num_frames'] for vm in video_metrics.values())
    weighted_miou = sum(vm['mIoU'] * vm['num_frames'] for vm in video_metrics.values()) / total_frames
    weighted_mnsd = sum(vm['mNSD'] * vm['num_frames'] for vm in video_metrics.values()) / total_frames
    weighted_final = sum(vm['Final_Score'] * vm['num_frames'] for vm in video_metrics.values()) / total_frames
    weighted_dice = sum(vm['Dice'] * vm['num_frames'] for vm in video_metrics.values()) / total_frames
    
    # Calculate per-class metrics across all videos (weighted by number of frames)
    weighted_miou_per_class = np.zeros(args.num_classes)
    weighted_mnsd_per_class = np.zeros(args.num_classes)
    weighted_final_per_class = np.zeros(args.num_classes)
    weighted_dice_per_class = np.zeros(args.num_classes)
    
    for video_name, vm in video_metrics.items():
        weighted_miou_per_class += np.array(vm['mIoU_per_class']) * vm['num_frames']
        weighted_mnsd_per_class += np.array(vm['mNSD_per_class']) * vm['num_frames']
        weighted_final_per_class += np.array(vm['Final_Score_per_class']) * vm['num_frames']
        weighted_dice_per_class += np.array(vm['Dice_per_class']) * vm['num_frames']
    
    weighted_miou_per_class /= total_frames
    weighted_mnsd_per_class /= total_frames
    weighted_final_per_class /= total_frames
    weighted_dice_per_class /= total_frames
    
    # Print overall results
    print("\n=== Overall Results (Weighted by Frame Count) ===")
    print(f"Overall mIoU: {weighted_miou:.4f}")
    print(f"Overall mNSD: {weighted_mnsd:.4f}")
    print(f"Overall Final Score: {weighted_final:.4f}")
    print(f"Overall Dice: {weighted_dice:.4f}")
    
    # Print per-class results
    print("\n=== Per-Class Results (Weighted by Frame Count) ===")
    for i in range(args.num_classes):
        class_name = class_names[i+1]  # Skip background
        print(f"Class {i+1} - {class_name}:")
        print(f"  mIoU: {weighted_miou_per_class[i]:.4f}")
        print(f"  mNSD: {weighted_mnsd_per_class[i]:.4f}")
        print(f"  Final Score: {weighted_final_per_class[i]:.4f}")
        print(f"  Dice: {weighted_dice_per_class[i]:.4f}")
    
    # Find best and worst performing videos
    best_video = max(video_metrics.items(), key=lambda x: x[1]['Final_Score'])
    worst_video = min(video_metrics.items(), key=lambda x: x[1]['Final_Score'])
    
    print(f"\nBest performing video: {best_video[0]} with Final Score: {best_video[1]['Final_Score']:.4f}")
    print(f"Worst performing video: {worst_video[0]} with Final Score: {worst_video[1]['Final_Score']:.4f}")
    
    # Save metrics to CSV
    import pandas as pd
    
    # Save overall video metrics
    metrics_df = pd.DataFrame.from_dict(video_metrics, orient='index')
    metrics_df.to_csv(os.path.join(args.output_dir, 'video_metrics.csv'))
    
    # Save per-class metrics for all videos
    per_class_metrics = {}
    
    # For each class
    for i in range(args.num_classes):
        class_name = class_names[i+1]  # Skip background
        class_metrics = {}
        
        # For each video
        for video_name, vm in video_metrics.items():
            class_metrics[video_name] = {
                'mIoU': vm['mIoU_per_class'][i],
                'mNSD': vm['mNSD_per_class'][i],
                'Final_Score': vm['Final_Score_per_class'][i],
                'Dice': vm['Dice_per_class'][i]
            }
        
        # Create DataFrame for this class
        class_df = pd.DataFrame.from_dict(class_metrics, orient='index')
        class_df.to_csv(os.path.join(args.output_dir, f'class_{i+1}_{class_name}_metrics.csv'))
        
        per_class_metrics[class_name] = {
            'mIoU': weighted_miou_per_class[i],
            'mNSD': weighted_mnsd_per_class[i],
            'Final_Score': weighted_final_per_class[i],
            'Dice': weighted_dice_per_class[i]
        }
    
    # Save overall per-class metrics
    overall_per_class_df = pd.DataFrame.from_dict(per_class_metrics, orient='index')
    overall_per_class_df.to_csv(os.path.join(args.output_dir, 'overall_per_class_metrics.csv'))
    
    # Save overall metrics to JSON
    overall_metrics = {
        'mIoU': float(weighted_miou),
        'mNSD': float(weighted_mnsd),
        'Final_Score': float(weighted_final),
        'Dice': float(weighted_dice),
        'mIoU_per_class': weighted_miou_per_class.tolist(),
        'mNSD_per_class': weighted_mnsd_per_class.tolist(),
        'Final_Score_per_class': weighted_final_per_class.tolist(),
        'Dice_per_class': weighted_dice_per_class.tolist()
    }
    
    import json
    with open(os.path.join(args.output_dir, 'overall_metrics.json'), 'w') as f:
        json.dump(overall_metrics, f, indent=4)
    
    # Create summary plots
    # Overall performance by video
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    
    # Sort videos by performance for better visualization
    sorted_videos = sorted(video_metrics.items(), key=lambda x: x[1]['Final_Score'], reverse=True)
    video_names = [v[0] for v in sorted_videos]
    miou_values = [v[1]['mIoU'] for v in sorted_videos]
    mnsd_values = [v[1]['mNSD'] for v in sorted_videos]
    final_values = [v[1]['Final_Score'] for v in sorted_videos]
    dice_values = [v[1]['Dice'] for v in sorted_videos]
    
    # Plot mIoU
    axes[0, 0].bar(video_names, miou_values, color='skyblue')
    axes[0, 0].set_title('mIoU by Video')
    axes[0, 0].set_ylabel('mIoU')
    axes[0, 0].set_xticklabels(video_names, rotation=90)
    axes[0, 0].axhline(y=weighted_miou, color='r', linestyle='-', label=f'Average: {weighted_miou:.4f}')
    axes[0, 0].legend()
    
    # Plot mNSD
    axes[0, 1].bar(video_names, mnsd_values, color='lightgreen')
    axes[0, 1].set_title('mNSD by Video')
    axes[0, 1].set_ylabel('mNSD')
    axes[0, 1].set_xticklabels(video_names, rotation=90)
    axes[0, 1].axhline(y=weighted_mnsd, color='r', linestyle='-', label=f'Average: {weighted_mnsd:.4f}')
    axes[0, 1].legend()
    
    # Plot Final Score
    axes[1, 0].bar(video_names, final_values, color='coral')
    axes[1, 0].set_title('Final Score by Video')
    axes[1, 0].set_ylabel('Final Score')
    axes[1, 0].set_xticklabels(video_names, rotation=90)
    axes[1, 0].axhline(y=weighted_final, color='r', linestyle='-', label=f'Average: {weighted_final:.4f}')
    axes[1, 0].legend()
    
    # Plot Dice
    axes[1, 1].bar(video_names, dice_values, color='plum')
    axes[1, 1].set_title('Dice Score by Video')
    axes[1, 1].set_ylabel('Dice Score')
    axes[1, 1].set_xticklabels(video_names, rotation=90)
    axes[1, 1].axhline(y=weighted_dice, color='r', linestyle='-', label=f'Average: {weighted_dice:.4f}')
    axes[1, 1].legend()
    
    plt.tight_layout()
    plt.savefig(os.path.join(args.output_dir, 'performance_by_video.png'), dpi=200)
    plt.close()
    
    # Create overall per-class metrics visualization
    class_names_no_bg = class_names[1:]  # Skip background
    
    fig, axes = plt.subplots(2, 2, figsize=(16, 14))
    
    # Plot overall mIoU per class
    axes[0, 0].bar(class_names_no_bg, weighted_miou_per_class, color='skyblue')
    axes[0, 0].set_title('Overall mIoU per Class', fontsize=14)
    axes[0, 0].set_ylabel('mIoU')
    axes[0, 0].set_ylim(0, 1.0)
    for i, v in enumerate(weighted_miou_per_class):
        axes[0, 0].text(i, v + 0.02, f'{v:.3f}', ha='center', fontsize=10)
    axes[0, 0].axhline(y=weighted_miou, color='r', linestyle='-', label=f'Average: {weighted_miou:.3f}')
    axes[0, 0].legend()
    plt.setp(axes[0, 0].xaxis.get_majorticklabels(), rotation=45, ha='right')
    
    # Plot overall mNSD per class
    axes[0, 1].bar(class_names_no_bg, weighted_mnsd_per_class, color='lightgreen')
    axes[0, 1].set_title('Overall mNSD per Class', fontsize=14)
    axes[0, 1].set_ylabel('mNSD')
    axes[0, 1].set_ylim(0, 1.0)
    for i, v in enumerate(weighted_mnsd_per_class):
        axes[0, 1].text(i, v + 0.02, f'{v:.3f}', ha='center', fontsize=10)
    axes[0, 1].axhline(y=weighted_mnsd, color='r', linestyle='-', label=f'Average: {weighted_mnsd:.3f}')
    axes[0, 1].legend()
    plt.setp(axes[0, 1].xaxis.get_majorticklabels(), rotation=45, ha='right')
    
    # Plot overall Final Score per class
    axes[1, 0].bar(class_names_no_bg, weighted_final_per_class, color='coral')
    axes[1, 0].set_title('Overall Final Score per Class', fontsize=14)
    axes[1, 0].set_ylabel('Final Score')
    axes[1, 0].set_ylim(0, 1.0)
    for i, v in enumerate(weighted_final_per_class):
        axes[1, 0].text(i, v + 0.02, f'{v:.3f}', ha='center', fontsize=10)
    axes[1, 0].axhline(y=weighted_final, color='r', linestyle='-', label=f'Average: {weighted_final:.3f}')
    axes[1, 0].legend()
    plt.setp(axes[1, 0].xaxis.get_majorticklabels(), rotation=45, ha='right')
    
    # Plot overall Dice per class
    axes[1, 1].bar(class_names_no_bg, weighted_dice_per_class, color='plum')
    axes[1, 1].set_title('Overall Dice Score per Class', fontsize=14)
    axes[1, 1].set_ylabel('Dice Score')
    axes[1, 1].set_ylim(0, 1.0)
    for i, v in enumerate(weighted_dice_per_class):
        axes[1, 1].text(i, v + 0.02, f'{v:.3f}', ha='center', fontsize=10)
    axes[1, 1].axhline(y=weighted_dice, color='r', linestyle='-', label=f'Average: {weighted_dice:.3f}')
    axes[1, 1].legend()
    plt.setp(axes[1, 1].xaxis.get_majorticklabels(), rotation=45, ha='right')
    
    plt.tight_layout()
    plt.savefig(os.path.join(args.output_dir, 'overall_per_class_metrics.png'), dpi=200)
    plt.close()
    
    # Create heatmap of per-class performance across videos
    for metric_name in ['mIoU', 'mNSD', 'Final_Score', 'Dice']:
        plt.figure(figsize=(14, 10))
        
        # Create data matrix for heatmap: videos x classes
        heatmap_data = []
        for video_name in video_names:  # Use sorted videos
            video_metric = video_metrics[video_name]
            if metric_name == 'Dice':
                heatmap_data.append(video_metric['Dice_per_class'])
            else:
                metric_key = f'{metric_name}_per_class'
                heatmap_data.append(video_metric[metric_key])
        
        # Convert to numpy array for heatmap
        heatmap_array = np.array(heatmap_data)
        
        # Create heatmap
        ax = sns.heatmap(heatmap_array, annot=True, cmap='viridis', fmt='.2f',
                       xticklabels=class_names_no_bg, yticklabels=video_names,
                       vmin=0, vmax=1.0, cbar_kws={'label': metric_name})
        
        plt.title(f'{metric_name} per Class across Videos', fontsize=16)
        plt.ylabel('Video', fontsize=12)
        plt.xlabel('Class', fontsize=12)
        
        # Rotate x-labels for better visibility
        plt.xticks(rotation=45, ha='right')
        plt.yticks(rotation=0)
        
        plt.tight_layout()
        plt.savefig(os.path.join(args.output_dir, f'{metric_name}_heatmap.png'), dpi=200, bbox_inches='tight')
        plt.close()
    
    # Find best and worst performing classes
    best_class_idx = np.argmax(weighted_final_per_class)
    worst_class_idx = np.argmin(weighted_final_per_class)
    
    print(f"\nBest performing class: {class_names_no_bg[best_class_idx]} with Final Score: {weighted_final_per_class[best_class_idx]:.4f}")
    print(f"Worst performing class: {class_names_no_bg[worst_class_idx]} with Final Score: {weighted_final_per_class[worst_class_idx]:.4f}")
    
    print(f"\nResults saved to {args.output_dir}")

if __name__ == '__main__':
    main()

