import tensorflow as tf

# Dice coefficient: A metric to measure the similarity between predicted and ground truth masks
def dice_coef(y_true, y_pred, smooth=1):
    # Flatten the tensors to calculate the Dice coefficient
    y_true_f = tf.keras.backend.flatten(y_true)
    y_pred_f = tf.keras.backend.flatten(y_pred)
    # Calculate the intersection and Dice coefficient
    intersection = tf.keras.backend.sum(y_true_f * y_pred_f)
    return (2. * intersection + smooth) / (tf.keras.backend.sum(y_true_f) + tf.keras.backend.sum(y_pred_f) + smooth)

# Dice loss: A loss function derived from the Dice coefficient
def dice_loss(y_true, y_pred, smooth=1):
    return 1 - dice_coef(y_true, y_pred, smooth)

# Boundary loss: Measures the difference in edges (boundaries) between predicted and ground truth masks
def boundary_loss(y_true, y_pred):
    # Add a channel dimension to the tensors
    y_true = tf.expand_dims(y_true, axis=-1)
    y_pred = tf.expand_dims(y_pred, axis=-1)
    # Compute the Sobel gradients (edges) for both ground truth and predicted masks
    gradient_true = tf.image.sobel_edges(y_true)
    gradient_pred = tf.image.sobel_edges(y_pred)
    # Calculate the mean absolute difference between the gradients
    return tf.reduce_mean(tf.abs(gradient_true - gradient_pred))

# Combined validation loss: Combines Dice loss and boundary loss with weighting factors
def combined_validation_loss(y_true, y_pred):
    dice = dice_loss(y_true, y_pred)  # Compute Dice loss
    boundary = boundary_loss(y_true, y_pred)  # Compute boundary loss
    # Combine the losses with 90% Dice loss and 10% boundary loss
    return 0.9 * dice + 0.1 * boundary
