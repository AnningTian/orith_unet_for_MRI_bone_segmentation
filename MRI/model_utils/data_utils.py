import glob
import cv2
import numpy as np
import random
import matplotlib.pyplot as plt

# Preview images and masks
def preview_image_and_mask(image_paths, mask_paths, target_size=(256, 256)):
    # Randomly select an index from the image and mask paths
    idx = random.randint(0, len(image_paths) - 1)
    image_path = image_paths[idx]
    mask_path = mask_paths[idx]

    # Load and normalize the RGB image
    image_rgb = plt.imread(image_path).astype(np.float32)
    if image_rgb.max() > 1.0:
        image_rgb /= 255.0  # Normalize pixel values to [0, 1]
    image_rgb = cv2.resize(image_rgb, target_size)  # Resize to the target size

    # Convert RGB image to grayscale and normalize
    image_gray = cv2.cvtColor((image_rgb * 255).astype(np.uint8), cv2.COLOR_RGB2GRAY).astype(np.float32) / 255.0
    image = np.dstack((image_rgb, image_gray))  # Stack RGB and grayscale as separate channels

    # Load and preprocess the mask
    mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE).astype(np.float32)
    if mask.max() > 1.0:
        mask /= 255.0  # Normalize pixel values to [0, 1]
    mask = cv2.resize(mask, target_size)  # Resize mask to target size
    mask = (mask > 0.5).astype(np.float32)  # Binarize mask (threshold at 0.5)

    # Plot the image and mask for preview
    fig, axes = plt.subplots(1, 2, figsize=(10, 5))
    axes[0].imshow((image_rgb * 255).astype(np.uint8))  # Display the RGB image
    axes[0].set_title('Preview Image (RGB)')
    axes[0].axis('off')

    axes[1].imshow((mask * 255).astype(np.uint8), cmap='gray')  # Display the mask
    axes[1].set_title('Preview Mask')
    axes[1].axis('off')

    plt.tight_layout()
    plt.show()

# Load and prepare images and masks for training
def load_and_prepare_data(image_paths, mask_paths, target_size=(256, 256)):
    # Initialize empty lists to store processed images and masks
    image_list, mask_list = [], []
    
    # Iterate through each pair of image and mask paths
    for image_path, mask_path in zip(image_paths, mask_paths):
        # Load and normalize the RGB image
        image_rgb = plt.imread(image_path).astype(np.float32)
        if image_rgb.max() > 1.0:
            image_rgb /= 255.0  # Normalize pixel values to [0, 1]
        image_rgb = cv2.resize(image_rgb, target_size)  # Resize to the target size

        # Convert RGB image to grayscale and normalize
        image_gray = cv2.cvtColor((image_rgb * 255).astype(np.uint8), cv2.COLOR_RGB2GRAY).astype(np.float32) / 255.0
        image = np.expand_dims(image_gray, axis=-1)  # Add a channel dimension for grayscale image

        # Load and preprocess the mask
        mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE).astype(np.float32)
        if mask.max() > 1.0:
            mask /= 255.0  # Normalize pixel values to [0, 1]
        mask = cv2.resize(mask, target_size)  # Resize mask to target size
        mask = (mask > 0.5).astype(np.float32)  # Binarize mask (threshold at 0.5)

        # Append processed image and mask to the respective lists
        image_list.append(image)
        mask_list.append(mask)

    # Convert lists to NumPy arrays for use in training
    return np.array(image_list), np.array(mask_list)
