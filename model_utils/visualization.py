import matplotlib.pyplot as plt
import numpy as np
import random

# Function to plot the training and validation loss over epochs
def plot_loss(history):
    plt.figure(figsize=(8, 6))
    # Plot training loss
    plt.plot(history.history['loss'], label='Train Loss')
    # Plot validation loss
    plt.plot(history.history['val_loss'], label='Validation Loss')
    plt.xlabel('Epochs')  # Label for the x-axis
    plt.ylabel('Loss')    # Label for the y-axis
    plt.title('Train and Validation Loss')  # Title of the plot
    plt.legend()  # Display legend
    plt.show()  # Show the plot

# Function to calculate the Intersection over Union (IoU) score
def calculate_iou(y_true, y_pred):
    # Compute the intersection and union of the binary masks
    intersection = np.logical_and(y_true, y_pred)
    union = np.logical_or(y_true, y_pred)
    # Calculate the IoU score
    return np.sum(intersection) / np.sum(union)

# Function to display model predictions alongside ground truth
def display_predictions(model, X_test, y_test):
    # Create a 3x3 grid of subplots
    fig, axes = plt.subplots(3, 3, figsize=(7, 7))

    for i in range(3):
        # Randomly select an index from the test data
        rand_idx = random.randint(0, len(X_test) - 1)
        augmented_img = X_test[rand_idx]  # Test image
        original_mask = y_test[rand_idx]  # Ground truth mask

        # Predict the mask using the model
        predicted_mask = model.predict(np.expand_dims(augmented_img, axis=0))
        # Binarize the predicted mask (threshold at 0.5)
        predicted_mask = (predicted_mask[0, :, :, 0] > 0.5).astype(int)

        # Calculate the IoU score for the predicted mask
        iou_score = calculate_iou(original_mask.squeeze(), predicted_mask)

        # Prepare images for visualization
        augmented_img_vis = (augmented_img[:, :, :3] * 255).astype(np.uint8)  # Convert test image to RGB format
        original_mask_vis = (original_mask * 255).astype(np.uint8)  # Convert ground truth mask to 8-bit format
        predicted_mask_vis = (predicted_mask * 255).astype(np.uint8)  # Convert predicted mask to 8-bit format

        # Display test image
        axes[i, 0].imshow(augmented_img_vis)
        axes[i, 0].set_title('Test Image')

        # Display ground truth mask
        axes[i, 1].imshow(original_mask_vis, cmap='gray')
        axes[i, 1].set_title('Original Mask')

        # Display predicted mask with IoU score
        axes[i, 2].imshow(predicted_mask_vis, cmap='gray')
        axes[i, 2].set_title(f'Predicted Mask\nIoU: {iou_score:.2f}')

        # Remove axis ticks for better visualization
        for ax in axes[i]:
            ax.axis('off')

    plt.tight_layout()  # Adjust subplot layout for better spacing
    plt.show()  # Show the figure
