import glob
import numpy as np
from model_utils.config import image_path, mask_path
from model_utils.data_utils import preview_image_and_mask, load_and_prepare_data
from model_utils import orith_unet
from model_utils.losses import combined_validation_loss, dice_coef
from model_utils.main import train_model
from model_utils.visualization import plot_loss, display_predictions
from sklearn.model_selection import train_test_split
import tensorflow as tf
import os

gpus = tf.config.list_physical_devices('GPU')
if gpus:
    try:
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)
    except RuntimeError as e:
        print(e)

# Load image and mask paths
image_paths = sorted(glob.glob(image_path), key=lambda x: os.path.basename(x))
mask_paths = sorted(glob.glob(mask_path), key=lambda x: os.path.basename(x))

# Preview images and masks
preview_image_and_mask(image_paths, mask_paths)

# Load data
images, masks = load_and_prepare_data(image_paths, mask_paths)
X_train, X_test, y_train, y_test = train_test_split(images, masks, test_size=0.2, random_state=23)

# Build model
model = orith_unet(input_shape=(256, 256, 1), num_classes=1)
model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=1e-4),
              loss=combined_validation_loss, metrics=['accuracy', dice_coef])

# Train model
history = train_model(model, X_train, y_train, X_test, y_test)

# Visualize training results
plot_loss(history)
display_predictions(model, X_test, y_test)
