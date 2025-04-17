import os
import sys
import cv2
import logging
import random
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.backends.cudnn as cudnn
import torch.optim as optim
from torch.nn.modules.loss import CrossEntropyLoss
from segment_anything import sam_model_registry
from pathlib import Path
import argparse
import warnings
import monai
from collections import Counter
from medpy import metric
from scipy.ndimage import zoom
import torchvision.transforms as transforms
from torchvision.transforms import InterpolationMode

# Import custom modules
from dataloader import get_loaders
from models import LoRA_Sam_v0_v2

# Import utils
from utils import DiceLoss

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
            image_batch, label_batch, low_res_label_batch = image_batch.to(device), label_batch.to(device), low_res_label_batch.to(device)
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
    loss_ce = ce_loss(output, label_batch[:].long())
    loss_dice = dice_loss(output, label_batch, softmax=True)
    loss = (1 - args.dice_weight) * loss_ce + args.dice_weight * loss_dice
    return loss, loss_ce, loss_dice


def training_per_epoch(model, trainloader, optimizer, iter_num, ce_loss, dice_loss, multimask_output=True, args=None):
    model.train()
    loss_all = []

    for i_batch, sampled_batch in enumerate(trainloader):
        image_batch, label_batch, low_res_label_batch = sampled_batch['image'], sampled_batch['mask'], sampled_batch['low_res_label']
        image_batch, label_batch, low_res_label_batch = image_batch.to(device), label_batch.to(device), low_res_label_batch.to(device)
        batch_dict = {'image_batch': label_batch, 'label_batch': label_batch, 'low_res_label_batch': low_res_label_batch}
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
            image_batch, label_batch, low_res_label_batch = sampled_batch['image'], sampled_batch['mask'], sampled_batch['low_res_label']
            image_batch, label_batch, low_res_label_batch = image_batch.to(device), label_batch.to(device), low_res_label_batch.to(device)
            batch_dict = {'image_batch': image_batch, 'label_batch': label_batch, 'low_res_label_batch': low_res_label_batch}
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
    parser.add_argument('--data_path', type=str, default=r'/SAN/medic/Surgical_LLM_Agent/SAR')    
    parser.add_argument('--dataset', type=str, default='Synapse', help='Experiment name')
    parser.add_argument('--num_classes', type=int, default=9, help='Number of classes')  # exclude backgrounds
    parser.add_argument('--list_dir', type=str, default='./lists/lists_Synapse/', help='list_dir')
    parser.add_argument('--output_dir', type=str, default='weight_results')
    parser.add_argument('--output_file', type=str, default='Endo_best.pt')
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
    os.makedirs(args.output_dir, exist_ok=True)

    # Set seed for reproducibility
    seed_everything(args.seed)
    
    # Initialize device
    global device
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Using device: {device}")

    # Load model
    sam, img_embedding_size = sam_model_registry[args.vit_name](
        image_size=args.img_size,
        num_classes=args.num_classes,
        checkpoint=args.ckpt, 
        pixel_mean=[0, 0, 0],
        pixel_std=[1, 1, 1]
    )

    # Initialize LoRA model
    net = LoRA_Sam_v0_v2(sam, args.rank).to(device)
    multimask_output = True if args.num_classes > 1 else False

    # Get data loaders
    trainloader, testloader, train_dataset, test_dataset = get_loaders(args)
    print(f'Training on: {device}, train sample size: {len(train_dataset)}, test sample size: {len(test_dataset)}, batch: {args.batch_size}')

    # Initialize loss functions
    ce_loss = CrossEntropyLoss()
    dice_loss = DiceLoss(args.num_classes + 1)
    
    # Initialize optimizer
    b_lr = args.base_lr / args.warmup_period if args.warmup else args.base_lr
    optimizer = optim.AdamW(filter(lambda p: p.requires_grad, net.parameters()), lr=b_lr, betas=(0.9, 0.999), weight_decay=0.1)
    iter_num = 0

    # Training loop
    best_epoch, best_loss = 0.0, np.inf
    for epoch in range(args.max_epochs):
        loss_training, iter_num, lr_current = training_per_epoch(net, trainloader, optimizer, iter_num, ce_loss, dice_loss, multimask_output=multimask_output, args=args)
        loss_testing, dice = test_per_epoch(net, testloader, ce_loss, dice_loss, multimask_output=True, args=args)

        if loss_testing < best_loss:
            best_loss = loss_testing
            best_epoch = epoch
            net.save_lora_parameters(os.path.join(args.output_dir, args.output_file))

        print('--- Epoch {}/{}: Training loss = {:.4f}, Testing: [loss = {:.4f}, dice = {:.4f}], Best loss = {:.4f}, Best epoch = {}, lr = {:.6f}'.\
              format(epoch, args.max_epochs, loss_training, loss_testing, dice, best_loss, best_epoch, lr_current))

    # Evaluation
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
    main()