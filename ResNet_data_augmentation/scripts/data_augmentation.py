# data_augmentation.py
"""
This script demonstrates data augmentation using PyTorch.
It takes an original dataset stored locally (with subfolders per class),
applies random transformations to generate augmented images, and
combines both the original and augmented images into a single dataset.
"""

import os
import shutil
import math
from PIL import Image
import torchvision.transforms as transforms

# -------------------------------
# 1. Define dataset paths (local)
# -------------------------------
original_dataset = './data/valid'           # Original dataset (organized by class)
augmented_dataset = './data/test_augmented'   # Directory to store augmented images
combined_dataset = './data/combined_dataset'  # Combined dataset (original + augmented)

# Create directories if they don't exist
os.makedirs(augmented_dataset, exist_ok=True)
os.makedirs(combined_dataset, exist_ok=True)

# -------------------------------
# 2. Define augmentation transform
# -------------------------------
# Using RandomAffine to mimic rotation, translation, scaling, and shearing.
# Also using RandomHorizontalFlip.
augmentation_transform = transforms.Compose([
    transforms.RandomAffine(
        degrees=20,
        translate=(0.2, 0.2),
        scale=(0.8, 1.2),
        shear=20,
        resample=Image.BILINEAR,
        fillcolor=0
    ),
    transforms.RandomHorizontalFlip()
])

# -------------------------------
# 3. Generate augmented images per class
# -------------------------------
def generate_augmented_images_for_class(class_dir, save_dir, num_aug_per_class=1000):
    """
    Generate augmented images for a given class folder.
    :param class_dir: Directory with original images for one class.
    :param save_dir: Destination directory for augmented images for that class.
    :param num_aug_per_class: Total number of augmented images to generate.
    """
    os.makedirs(save_dir, exist_ok=True)
    
    # List all image files in the original class folder
    image_files = [f for f in os.listdir(class_dir) if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
    if not image_files:
        print(f"No images found in {class_dir}")
        return

    # Calculate number of augmented images to generate per original image (ceiling)
    num_images = len(image_files)
    aug_per_image = math.ceil(num_aug_per_class / num_images)
    
    count = 0
    for image_file in image_files:
        img_path = os.path.join(class_dir, image_file)
        try:
            with Image.open(img_path) as img:
                img = img.convert('RGB')
                for i in range(aug_per_image):
                    # Apply augmentation transform
                    aug_img = augmentation_transform(img)
                    # Create a filename for the augmented image
                    aug_filename = f"aug_{os.path.splitext(image_file)[0]}_{i}.jpeg"
                    aug_img.save(os.path.join(save_dir, aug_filename))
                    count += 1
                    if count >= num_aug_per_class:
                        break
                if count >= num_aug_per_class:
                    break
        except Exception as e:
            print(f"Error processing image {img_path}: {e}")
    print(f"Generated {count} augmented images in {save_dir}")

# Process each class folder in the original dataset
for class_name in os.listdir(original_dataset):
    original_class_dir = os.path.join(original_dataset, class_name)
    if os.path.isdir(original_class_dir):
        augmented_class_dir = os.path.join(augmented_dataset, class_name)
        print(f"Processing class '{class_name}'...")
        generate_augmented_images_for_class(original_class_dir, augmented_class_dir, num_aug_per_class=1000)

# -------------------------------
# 4. Combine original and augmented datasets
# -------------------------------
def copy_images(src_dir, dest_dir):
    """
    Copy all images from src_dir (organized by class) to dest_dir.
    """
    for class_name in os.listdir(src_dir):
        src_class_dir = os.path.join(src_dir, class_name)
        if os.path.isdir(src_class_dir):
            dest_class_dir = os.path.join(dest_dir, class_name)
            os.makedirs(dest_class_dir, exist_ok=True)
            for filename in os.listdir(src_class_dir):
                src_file = os.path.join(src_class_dir, filename)
                dest_file = os.path.join(dest_class_dir, filename)
                shutil.copy(src_file, dest_file)

# Copy images from the original dataset and the augmented dataset
copy_images(original_dataset, combined_dataset)
copy_images(augmented_dataset, combined_dataset)
print("Combined dataset created at:", combined_dataset)
