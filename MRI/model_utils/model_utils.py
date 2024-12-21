import tensorflow as tf
from tensorflow.keras.layers import (Conv2D, Conv2DTranspose, MaxPooling2D, 
                                     concatenate, Input, Activation, Add,
                                     BatchNormalization, multiply)
from tensorflow.keras.models import Model

# Attention block: Enhances features using attention mechanism
def attention_block(x, g, inter_channel):
    # Apply 1x1 convolution to input x
    theta_x = Conv2D(inter_channel, kernel_size=1, strides=1, padding='same')(x)
    # Apply 1x1 convolution to gating signal g
    phi_g = Conv2D(inter_channel, kernel_size=1, strides=1, padding='same')(g)
    # Add results and pass through ReLU activation
    f = Activation('relu')(Add()([theta_x, phi_g]))
    # Apply 1x1 convolution and sigmoid activation to get attention coefficients
    psi_f = Conv2D(1, kernel_size=1, strides=1, padding='same')(f)
    rate = Activation('sigmoid')(psi_f)
    # Multiply input x with attention coefficients
    att_x = multiply([x, rate])
    return att_x

# UNet with attention mechanism
def orith_unet(input_shape=(256, 256, 1), num_classes=1):
    # Input layer
    inputs = Input(input_shape)

    # Encoder block 1
    c1 = Conv2D(64, (7, 7), activation='relu', padding='same')(inputs)
    c1 = BatchNormalization()(c1)
    p1 = MaxPooling2D((2, 2))(c1)

    # Encoder block 2
    c2 = Conv2D(128, (5, 5), activation='relu', padding='same')(p1)
    c2 = BatchNormalization()(c2)
    p2 = MaxPooling2D((2, 2))(c2)

    # Encoder block 3
    c3 = Conv2D(256, (3, 3), activation='relu', padding='same')(p2)
    c3 = BatchNormalization()(c3)
    p3 = MaxPooling2D((2, 2))(c3)

    # Encoder block 4
    c4 = Conv2D(512, (3, 3), activation='relu', padding='same')(p3)
    c4 = BatchNormalization()(c4)
    p4 = MaxPooling2D((2, 2))(c4)

    # Bottleneck
    bn = Conv2D(1024, (3, 3), activation='relu', padding='same')(p4)
    bn = BatchNormalization()(bn)

    # Decoder block 1
    u5 = Conv2DTranspose(512, (3, 3), strides=(2, 2), padding='same')(bn)
    a5 = attention_block(c4, u5, 512)  # Attention mechanism
    u5 = concatenate([u5, a5])  # Concatenate with encoder features
    u5 = Conv2D(512, (3, 3), activation='relu', padding='same')(u5)
    u5 = BatchNormalization()(u5)

    # Decoder block 2
    u6 = Conv2DTranspose(256, (3, 3), strides=(2, 2), padding='same')(u5)
    a6 = attention_block(c3, u6, 256)  # Attention mechanism
    u6 = concatenate([u6, a6])  # Concatenate with encoder features
    u6 = Conv2D(256, (3, 3), activation='relu', padding='same')(u6)
    u6 = BatchNormalization()(u6)

    # Decoder block 3
    u7 = Conv2DTranspose(128, (3, 3), strides=(2, 2), padding='same')(u6)
    a7 = attention_block(c2, u7, 128)  # Attention mechanism
    u7 = concatenate([u7, a7])  # Concatenate with encoder features
    u7 = Conv2D(128, (3, 3), activation='relu', padding='same')(u7)
    u7 = BatchNormalization()(u7)

    # Decoder block 4
    u8 = Conv2DTranspose(64, (3, 3), strides=(2, 2), padding='same')(u7)
    a8 = attention_block(c1, u8, 64)  # Attention mechanism
    u8 = concatenate([u8, a8])  # Concatenate with encoder features
    u8 = Conv2D(64, (3, 3), activation='relu', padding='same')(u8)
    u8 = BatchNormalization()(u8)

    # Output layer: 1x1 convolution to generate final predictions
    outputs = Conv2D(num_classes, (1, 1), activation='sigmoid')(u8)

    # Create the model
    model = Model(inputs=[inputs], outputs=[outputs])
    return model
