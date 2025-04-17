#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import glob

def count_segmentation_images(dataset_dir):
    """
    Counts PNG images in 'segmentation' folders for each video folder and 
    saves the count to a text file in the respective video folder.
    Also calculates total frame counts for Train and Test datasets.
    
    Args:
        dataset_dir (str): Path to the root directory of the SAR dataset
    """
    # Dictionary to store total counts for each subset
    total_counts = {'Train': 0, 'Test': 0}
    
    # Process both Train and Test directories
    for subset in ['Train', 'Test']:
        subset_dir = os.path.join(dataset_dir, subset)
        subset_count = 0
        
        # Skip if directory doesn't exist
        if not os.path.isdir(subset_dir):
            print(f"Directory not found: {subset_dir}")
            continue
            
        print(f"Processing {subset} directory...")
        
        # Iterate through all video directories in the subset
        for video_dir in os.listdir(subset_dir):
            video_path = os.path.join(subset_dir, video_dir)
            
            # Skip if not a directory
            if not os.path.isdir(video_path):
                continue
                
            # Path to the segmentation folder
            segmentation_path = os.path.join(video_path, 'segmentation')
            
            # Skip if segmentation folder doesn't exist
            if not os.path.isdir(segmentation_path):
                print(f"Segmentation folder not found: {segmentation_path}")
                continue
                
            # Count PNG files in the segmentation folder
            png_files = glob.glob(os.path.join(segmentation_path, '*.png'))
            image_count = len(png_files)
            
            # Add to subset total
            subset_count += image_count
            
            # Write count to a text file in the video folder
            output_file = os.path.join(video_path, 'segmentation_count.txt')
            with open(output_file, 'w') as f:
                f.write(f"Number of segmentation images: {image_count}\n")
                
            print(f"Processed {video_dir}: {image_count} images")
        
        # Store total count for this subset
        total_counts[subset] = subset_count
        print(f"Total images in {subset}: {subset_count}")
    
    # Write total counts to a summary file in the dataset root directory
    summary_file = os.path.join(dataset_dir, 'segmentation_summary.txt')
    with open(summary_file, 'w') as f:
        f.write("Segmentation Images Summary\n")
        f.write("==========================\n\n")
        f.write(f"Train dataset total frames: {total_counts['Train']}\n")
        f.write(f"Test dataset total frames: {total_counts['Test']}\n")
        f.write(f"Total frames (Train + Test): {total_counts['Train'] + total_counts['Test']}\n")
    
    print("\nSummary:")
    print(f"Train dataset: {total_counts['Train']} frames")
    print(f"Test dataset: {total_counts['Test']} frames")
    print(f"Total: {total_counts['Train'] + total_counts['Test']} frames")
    print(f"Summary saved to: {summary_file}")
    print("Processing complete!")

if __name__ == "__main__":
    # Change this to your dataset root directory
    dataset_dir = r"D:\SAR-dataset"
    
    # If you want to use a command line argument instead
    import sys
    if len(sys.argv) > 1:
        dataset_dir = sys.argv[1]
    
    print(f"Counting segmentation images in {dataset_dir}...")
    count_segmentation_images(dataset_dir)