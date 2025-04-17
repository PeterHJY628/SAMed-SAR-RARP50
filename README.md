
# SAMed-SAR-RARP50 User Guide

This guide will help you set up the environment and run the project.

---

## 1. Clone the repository

```bash
git clone <repository_url>
cd SAMed-SAR-RARP50
```

## 2. Download the dataset

Download the training and test sets from:

- **Train set**: https://rdr.ucl.ac.uk/articles/dataset/SAR-RARP50_train_set/24932529
- **Test set**: https://rdr.ucl.ac.uk/articles/dataset/SAR-RARP50_test_set/24932499

Organize your videos into the following directory structure as needed:

```
data/
├── Train
├── Validate
└── Test
```

## 3. Frame extraction

Extract frames from videos at 60Hz using `Frame_extraction.py`:

```bash
python Frame_extraction.py --input_dir data/Train --output_dir your_path
```

## 4. Download pretrained weights

Switch to the `SAMed` directory and download the SAMed and LoRA weights with gdown:

```bash
cd SAMed

gdown 1P0Bm-05l-rfeghbrT1B62v5eN-3A-uOr  # SAMed weights

gdown 1_oCdoEEu3mNhRfFxeWyRerOKt8OEUvcg  # LoRA weights
```

## 5. Train the model with LoRA integration

Run the unified training script (e.g., `SAR_Train.py`) to train SAM with LoRA adaptation and perform evaluation:

```bash
python SAR_Train.py \
  --data_path ../data \              # Path to the root `data/` directory
  --dataset Synapse \               # Dataset identifier
  --num_classes 9 \                 # Number of segmentation classes (exclude background)
  --img_size 512 \                  # Input resolution for the image encoder
  --batch_size 64 \                 # Batch size per GPU (adjust to fit your memory)
  --max_epochs 100 \                # Total number of training epochs
  --base_lr 0.001 \                 # Base learning rate for segmentation network
  --ckpt checkpoints/sam_vit_b_01ec64.pth \  # Pretrained SAM checkpoint
  --lora_ckpt results/Endo_best.pt \        # Path to save/load LoRA weights
  --rank 6 \                        # LoRA rank hyperparameter
  --use_official_metrics \         # Compute official mIoU and mNSD after inference
  --save_predictions                # Save predicted masks for external evaluation
```

**Tips:**

- Adjust `--batch_size`, `--max_epochs`, and `--rank` according to your GPU resources.
- The script will automatically save the best LoRA parameters to the location specified by `--lora_ckpt`.
- Intermediate checkpoints, logs, and prediction outputs will be stored in `results/` by default.
## 6. Inference on Test Videos

Run the inference script `SAR_Inference.py` to segment every video in your test subset, save predictions, and compute metrics:

```bash
python SAR_Inference.py \
  --data_path ../data \               # Root data directory
  --subset Test \                     # Subdirectory with test videos
  --ckpt checkpoints/sam_vit_b_01ec64.pth \  # Path to SAM checkpoint
  --lora_ckpt results/Endo_best.pt \        # Path to saved LoRA weights
  --output_dir inference_results \    # Directory to store all outputs
  --batch_size 64 \                   # Batch size for inference
  --img_size 512 \                    # Input resolution for model
  --num_classes 9 \                   # Number of classes (excluding background)
  --vit_name vit_b \                  # ViT model variant
  --rank 6 \                          # LoRA rank
  [--keep_temp_files] \               # (Optional) retain intermediate temp files
  [--save_all_predictions]             # (Optional) save all frame predictions, not just best
```

**What this does:**

- **Per-video segmentation:** Iterates over each folder under `data_path/subset`, treating it as a separate video.  
- **Output structure:** Under `output_dir`, you will find subfolders:
  - `all_predictions/` & `all_originals/` & `all_ground_truth/` for raw and colorized masks at both original and high resolutions.  
  - `best_frames/` containing the best-performing frame per video (raw, colorized, combined visualization, and an `info.txt`).  
  - `metrics/` with per-video JSON files summarizing mIoU, mNSD, Final_Score, Dice and per-class lists.  
  - `per_class_metrics/` with aggregated per-class bar charts and overall summaries.  
  - CSV/JSON tables for video- and class-level metrics, plus summary plots under `performance_by_video.png` and `overall_per_class_metrics.png`.  
- **Metrics computed:**
  - **Per-frame & per-video** confusion matrix–based Dice scores.  
  - **Official** mIoU and mNSD (via MONAI), plus a combined Final Score (√(mIoU × mNSD)).  
  - **Per-class** and **overall** metrics, with weighted averages across videos.  
  - **Heatmaps** of per-class performance across all videos.

After running, the script will print best/worst performing videos and classes, and save all results in `inference_results/`.  

---

With these two scripts (`SAR_Train.py` and `SAR_Inference.py`), you can fully train and evaluate SAM with LoRA on the SAR-RARP50 dataset. Enjoy your experiments!

