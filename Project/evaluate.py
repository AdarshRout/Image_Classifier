import tensorflow as tf
from data_preprocessing import load_and_preprocess_data
from sklearn.metrics import precision_recall_fscore_support

# Load the trained model
model = tf.keras.models.load_model('models/cifar10_resnet_model.h5')

# Load and preprocess test data
_, (x_test, y_test) = load_and_preprocess_data()

# Evaluate the model
test_loss, test_accuracy = model.evaluate(x_test, y_test)
print(f"Test accuracy: {test_accuracy:.4f}")

# Get predictions
y_pred = model.predict(x_test)
y_pred_classes = tf.argmax(y_pred, axis=1)
y_true_classes = tf.argmax(y_test, axis=1)

# Calculate precision, recall, and F1-score
precision, recall, f1, _ = precision_recall_fscore_support(y_true_classes, y_pred_classes, average='weighted')

print(f"Precision: {precision:.4f}")
print(f"Recall: {recall:.4f}")
print(f"F1-score: {f1:.4f}")