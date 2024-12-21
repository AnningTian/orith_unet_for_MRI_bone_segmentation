import os
import cv2
import numpy as np
import tensorflow as tf
import random
import matplotlib.pyplot as plt
from sklearn.metrics import jaccard_score

# Load and prepare test dataset
def load_test_data(image_dir, mask_dir, target_size=(256, 256)):
    image_paths = sorted([os.path.join(image_dir, fname) for fname in os.listdir(image_dir) if fname.endswith('.png')])
    mask_paths = sorted([os.path.join(mask_dir, fname) for fname in os.listdir(mask_dir) if fname.endswith('.png')])

    original_sizes = []
    images, masks = [], []

    for image_path, mask_path in zip(image_paths, mask_paths):
        # Load and preprocess image
        image = plt.imread(image_path).astype(np.float32)
        if image.max() > 1.0:
            image /= 255.0
        original_size = image.shape[:2]
        original_sizes.append(original_size)
        image = cv2.resize(image, target_size)

        # Convert to grayscale and expand dims
        image_gray = cv2.cvtColor((image * 255).astype(np.uint8), cv2.COLOR_RGB2GRAY).astype(np.float32) / 255.0
        image_gray = np.expand_dims(image_gray, axis=-1)
        
        # Load and preprocess mask
        mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE).astype(np.float32)
        if mask.max() > 1.0:
            mask /= 255.0
        mask = cv2.resize(mask, target_size)
        mask = (mask > 0.5).astype(np.float32)

        images.append(image_gray)
        masks.append(mask)

    return np.array(images), np.array(masks), original_sizes

# Dice coefficient
def dice_coef(y_true, y_pred, smooth=1):
    y_true_f = y_true.flatten()
    y_pred_f = y_pred.flatten()
    intersection = np.sum(y_true_f * y_pred_f)
    return (2. * intersection + smooth) / (np.sum(y_true_f) + np.sum(y_pred_f) + smooth)

# Predict and evaluate
def evaluate_model(model, test_images, test_masks):
    dice_scores = []
    iou_scores = []

    for i in range(len(test_images)):
        image = np.expand_dims(test_images[i], axis=0)
        true_mask = test_masks[i]

        pred_mask = model.predict(image)[0, :, :, 0]
        pred_mask = (pred_mask > 0.5).astype(np.float32)

        if np.sum(true_mask) == 0:  # If ground truth is completely black
            if np.sum(pred_mask) == 0:
                dice = 1.0
                iou = 1.0
            else:
                dice = 0.0
                iou = 0.0
        else:
            dice = dice_coef(true_mask, pred_mask)
            iou = jaccard_score(true_mask.flatten(), pred_mask.flatten())

        dice_scores.append(dice)
        iou_scores.append(iou)

    return dice_scores, iou_scores

# Display predictions
def display_random_predictions(model, test_images, test_masks, original_sizes, num_samples=3):
    indices = random.sample(range(len(test_images)), num_samples)

    fig, axes = plt.subplots(num_samples, 3, figsize=(10, num_samples * 3))
    for i, idx in enumerate(indices):
        image = test_images[idx]
        true_mask = test_masks[idx]
        original_size = original_sizes[idx]
        pred_mask = model.predict(np.expand_dims(image, axis=0))[0, :, :, 0]
        pred_mask = (pred_mask > 0.5).astype(np.float32)

        # Resize to original size
        resized_image = cv2.resize(image.squeeze(), (original_size[1], original_size[0]))
        resized_true_mask = cv2.resize(true_mask, (original_size[1], original_size[0]))
        resized_pred_mask = cv2.resize(pred_mask, (original_size[1], original_size[0]))

        axes[i, 0].imshow(resized_image, cmap='gray')
        axes[i, 0].set_title('Test Image')
        axes[i, 0].axis('off')

        axes[i, 1].imshow(resized_true_mask, cmap='gray')
        axes[i, 1].set_title('Original Mask')
        axes[i, 1].axis('off')

        axes[i, 2].imshow(resized_pred_mask, cmap='gray')
        axes[i, 2].set_title('Predicted Mask')
        axes[i, 2].axis('off')

    plt.tight_layout()
    plt.show()

# Main function
if __name__ == "__main__":
    test_img_dir = input("Enter the test image directory: ")
    test_mask_dir = input("Enter the test mask directory: ")
    model_path = input("Enter the path to the trained model (.h5 file): ")

    # Load model
    model = tf.keras.models.load_model(model_path, custom_objects={'combined_validation_loss': None, 'dice_coef': dice_coef})

    # Load test data
    test_images, test_masks, original_sizes = load_test_data(test_img_dir, test_mask_dir)

    # Evaluate
    dice_scores, iou_scores = evaluate_model(model, test_images, test_masks)
    print(f"Overall Dice Score: {np.mean(dice_scores):.4f}")
    print(f"Overall IoU Score: {np.mean(iou_scores):.4f}")

    # Display predictions
    display_random_predictions(model, test_images, test_masks, original_sizes)
