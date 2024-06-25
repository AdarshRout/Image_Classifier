import tensorflow as tf
from data_preprocessing import load_and_preprocess_data, data_augmentation
from model import create_resnet_model

# Load and preprocess data
(x_train, y_train), (x_test, y_test) = load_and_preprocess_data()

# Create model
model = create_resnet_model()

# Compile model
optimizer = tf.keras.optimizers.Adam(learning_rate=0.001)
model.compile(optimizer=optimizer,
              loss='categorical_crossentropy',
              metrics=['accuracy'])

# Data augmentation
data_augmentation_layer = data_augmentation()

# Early stopping
early_stopping = tf.keras.callbacks.EarlyStopping(
    monitor='val_accuracy',
    patience=10,
    restore_best_weights=True
)

# Train the model
history = model.fit(
    data_augmentation_layer(x_train),
    y_train,
    batch_size=64,
    epochs=100,
    validation_split=0.1,
    callbacks=[early_stopping]
)

# Save the model
model.save('models/cifar10_resnet_model.h5')

# Plot training history
import matplotlib.pyplot as plt

plt.figure(figsize=(12, 4))
plt.subplot(1, 2, 1)
plt.plot(history.history['accuracy'], label='Training Accuracy')
plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
plt.title('Model Accuracy')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.legend()

plt.subplot(1, 2, 2)
plt.plot(history.history['loss'], label='Training Loss')
plt.plot(history.history['val_loss'], label='Validation Loss')
plt.title('Model Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend()

plt.tight_layout()
plt.savefig('static/training_history.png')