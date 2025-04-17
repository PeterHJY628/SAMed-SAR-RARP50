#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import glob
import cv2
import re
import numpy as np

def extract_matching_frames(dataset_dir):
    """
    Extracts frames from video files that correspond to segmentation images.
    
    Args:
        dataset_dir (str): Path to the root directory of the SAR dataset
    """
    # Process both Train and Test directories
    for subset in ['Train', 'Test']:
        subset_dir = os.path.join(dataset_dir, subset)
        
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
            
            # Look for the video file (expecting video_left.avi as per documentation)
            video_file = os.path.join(video_path, 'video_left.avi')
            if not os.path.isfile(video_file):
                print(f"Video file not found: {video_file}")
                continue
                
            print(f"Processing video: {video_dir}")
            
            # Create output directory for extracted frames
            frames_dir = os.path.join(video_path, 'images')
            if not os.path.exists(frames_dir):
                os.makedirs(frames_dir)
                print(f"  Created images directory: {frames_dir}")
            
            # Get list of segmentation files
            segmentation_files = glob.glob(os.path.join(segmentation_path, '*.png'))
            frame_numbers = []
            
            # Extract frame numbers from segmentation filenames
            for seg_file in segmentation_files:
                # Extract frame number from filename (format: 000000060.png)
                base_name = os.path.basename(seg_file)
                frame_num = int(os.path.splitext(base_name)[0])
                frame_numbers.append(frame_num)
                
            # Print information about frame 0
            if 0 in frame_numbers:
                print("  Frame 0 found in segmentation files and will be extracted")
            else:
                print("  Note: Frame 0 not found in segmentation files")
            
            # Sort frame numbers
            frame_numbers.sort()
            
            # Open the video file
            cap = cv2.VideoCapture(video_file)
            if not cap.isOpened():
                print(f"Error: Could not open video {video_file}")
                continue
                
            total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            print(f"Total frames in video: {total_frames}")
            print(f"Extracting {len(frame_numbers)} frames...")
            
            # Process each frame number
            for frame_num in frame_numbers:
                # Set video position to the frame
                cap.set(cv2.CAP_PROP_POS_FRAMES, frame_num)
                
                # Read the frame
                ret, frame = cap.read()
                if not ret:
                    print(f"  Warning: Could not read frame {frame_num}")
                    continue
                    
                # Save the frame
                frame_filename = os.path.join(frames_dir, f"{frame_num:09d}.png")
                cv2.imwrite(frame_filename, frame)
                
                # Print confirmation for frame 0 to ensure it's being processed
                if frame_num == 0:
                    print(f"  Successfully extracted frame 0")
            
            # Release the video
            cap.release()
            
            print(f"  Extracted {len(frame_numbers)} frames to {frames_dir} folder")
            
    print("Processing complete!")

if __name__ == "__main__":
    # Change this to your dataset root directory
    dataset_dir = r"D:\SAR-dataset"
    
    # If you want to use a command line argument instead
    import sys
    if len(sys.argv) > 1:
        dataset_dir = sys.argv[1]
    
    print(f"Extracting matching frames in {dataset_dir}...")
    extract_matching_frames(dataset_dir)