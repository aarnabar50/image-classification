#!/usr/bin/env python3
"""
Create a subset of images from a source directory.
Takes a percentage of images from each class folder.
"""

import os
import shutil
import random
from pathlib import Path

def create_subset(source_dir, target_dir, percentage=10):
    """
    Create a subset by copying percentage of images from each class.
    
    Args:
        source_dir: Source directory with class subdirectories
        target_dir: Target directory to create subset
        percentage: Percentage of images to copy from each class (1-100)
    """
    source_path = Path(source_dir)
    target_path = Path(target_dir)
    
    # Create target directory
    target_path.mkdir(parents=True, exist_ok=True)
    
    total_source_images = 0
    total_copied_images = 0
    
    # Get all class directories
    class_dirs = [d for d in source_path.iterdir() if d.is_dir() and not d.name.startswith('.')]
    
    print(f"\n{'='*60}")
    print(f"Creating {percentage}% subset from {source_dir}")
    print(f"Target directory: {target_dir}")
    print(f"{'='*60}\n")
    
    for class_dir in sorted(class_dirs):
        class_name = class_dir.name
        
        # Get all image files in this class
        image_extensions = {'.jpg', '.jpeg', '.png', '.bmp', '.gif'}
        all_images = [f for f in class_dir.iterdir() 
                      if f.is_file() and f.suffix.lower() in image_extensions]
        
        total_source_images += len(all_images)
        
        # Calculate number of images to copy
        num_to_copy = max(1, int(len(all_images) * percentage / 100))
        
        # Randomly sample images
        random.seed(42)  # For reproducibility
        selected_images = random.sample(all_images, num_to_copy)
        
        # Create target class directory
        target_class_dir = target_path / class_name
        target_class_dir.mkdir(parents=True, exist_ok=True)
        
        # Copy selected images
        for img_path in selected_images:
            target_img_path = target_class_dir / img_path.name
            shutil.copy2(img_path, target_img_path)
            total_copied_images += 1
        
        print(f"Class '{class_name}': {len(all_images)} images -> copied {num_to_copy} ({num_to_copy/len(all_images)*100:.1f}%)")
    
    actual_percentage = (total_copied_images / total_source_images * 100) if total_source_images > 0 else 0
    
    print(f"\n{'='*60}")
    print(f"SUBSET CREATION COMPLETE")
    print(f"{'='*60}")
    print(f"Total source images: {total_source_images}")
    print(f"Total copied images: {total_copied_images}")
    print(f"Actual percentage: {actual_percentage:.2f}%")
    print(f"{'='*60}\n")


if __name__ == "__main__":
    # Configuration
    SOURCE_DIR = "/Users/aarnabar/image-classification/data/cropped_lisa_1/train_1"
    TARGET_DIR = "/Users/aarnabar/image-classification/data/cropped_lisa_1/train_1_subset_10pct"
    PERCENTAGE = 10  # 10% of images from each class
    
    create_subset(SOURCE_DIR, TARGET_DIR, PERCENTAGE)
    
    print(f"\nSubset created successfully!")
    print(f"You can now use this directory for adversarial image generation:")
    print(f"  {TARGET_DIR}")
