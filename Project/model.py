import tensorflow as tf
from tensorflow.keras import layers, models

def create_resnet_block(inputs, filters, kernel_size=3, stride=1, conv_shortcut=False):
    x = layers.Conv2D(filters, kernel_size, strides=stride, padding='same')(inputs)
    x = layers.BatchNormalization()(x)
    x = layers.Activation('relu')(x)
    
    x = layers.Conv2D(filters, kernel_size, padding='same')(x)
    x = layers.BatchNormalization()(x)
    
    if conv_shortcut:
        shortcut = layers.Conv2D(filters, 1, strides=stride, padding='same')(inputs)
        shortcut = layers.BatchNormalization()(shortcut)
    else:
        shortcut = inputs
    
    x = layers.Add()([x, shortcut])
    x = layers.Activation('relu')(x)
    return x

def create_resnet_model():
    inputs = layers.Input(shape=(32, 32, 3))
    
    x = layers.Conv2D(64, 3, strides=1, padding='same')(inputs)
    x = layers.BatchNormalization()(x)
    x = layers.Activation('relu')(x)
    
    x = create_resnet_block(x, 64, conv_shortcut=True)
    x = create_resnet_block(x, 64)
    
    x = create_resnet_block(x, 128, stride=2, conv_shortcut=True)
    x = create_resnet_block(x, 128)
    
    x = create_resnet_block(x, 256, stride=2, conv_shortcut=True)
    x = create_resnet_block(x, 256)
    
    x = layers.GlobalAveragePooling2D()(x)
    x = layers.Dense(1024, activation='relu')(x)
    x = layers.Dropout(0.5)(x)
    outputs = layers.Dense(10, activation='softmax')(x)
    
    model = models.Model(inputs, outputs)
    return model