import tensorflow as tf
from tensorflow.keras.callbacks import ReduceLROnPlateau, EarlyStopping, ModelCheckpoint
from sklearn.model_selection import train_test_split

def train_model(model, X_train, y_train, X_test, y_test, batch_size=4, epochs=25):
    """
    Train the given model with provided training and validation data.
    
    Parameters:
        model: tf.keras.Model
            The compiled Keras model to train.
        X_train: np.ndarray
            Training images.
        y_train: np.ndarray
            Training masks.
        X_test: np.ndarray
            Validation images.
        y_test: np.ndarray
            Validation masks.
        batch_size: int, optional
            Batch size for training. Default is 4.
        epochs: int, optional
            Number of epochs to train the model. Default is 25.

    Returns:
        history: keras.callbacks.History
            Training history object containing details of the training process.
    """
    # Define callbacks
    model_checkpoint_best = ModelCheckpoint('models/unet_mri_best_v5.h5', monitor='val_loss', save_best_only=True, mode='min')
    model_checkpoint_last = ModelCheckpoint('models/unet_mri_final_v5.h5', save_best_only=False)

    reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=3, min_lr=1e-6)
    early_stopping = EarlyStopping(monitor='val_loss', patience=30, restore_best_weights=True)

    # Train the model
    history = model.fit(
        X_train,
        y_train,
        batch_size=batch_size,
        epochs=epochs,
        validation_data=(X_test, y_test),
        callbacks=[reduce_lr, early_stopping, model_checkpoint_best, model_checkpoint_last]
    )
    
    return history
