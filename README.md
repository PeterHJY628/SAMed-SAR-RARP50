
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
python Frame_extraction.py --input_dir data/Train --output_dir frames/Train --fps 60
python Frame_extraction.py --input_dir data/Test --output_dir frames/Test --fps 60
```

## 4. Download pretrained weights

Switch to the `SAMed` directory and download the SAMed and LoRA weights with gdown:

```bash
cd SAMed

gdown 1P0Bm-05l-rfeghbrT1B62v5eN-3A-uOr  # SAMed weights

gdown 1_oCdoEEu3mNhRfFxeWyRerOKt8OEUvcg  # LoRA weights
```

## 5. Train the model

Run the training script `SAR_Train.py`:

```bash
python SAR_Train.py \
  --train_dir ../frames/Train \
  --val_dir ../frames/Validate \
  --output_dir checkpoints \
  --batch_size 16 \
  --epochs 50
```

*Tip:* Adjust `--batch_size` according to your GPU memory and set `--epochs` to control the number of training epochs.

## 6. Inference & evaluation

After training, run `SAR_Inference.py` to perform inference and evaluate performance:

```bash
python SAR_Inference.py \
  --model_path checkpoints/best_model.pth \
  --test_dir ../frames/Test \
  --output_dir results \
  --metrics iou precision recall
```

The script will save predictions and compute the specified metrics.

---

If you encounter any issues or have suggestions, please open an issue on the repository. Enjoy!
```

