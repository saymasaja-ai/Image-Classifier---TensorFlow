import tensorflow as tf
import tensorflow_datasets as tfds
import matplotlib.pyplot as plt
import json

from tensorflow.python.keras import models,layers
import tensorflow_hub as hub
from tensorflow.python.keras.models import load_model


# TensorFlow 2.20 bug patch
import tensorflow.python.distribute.input_lib as input_lib
if not hasattr(input_lib, "DistributedDatasetInterface"):
    input_lib.DistributedDatasetInterface = input_lib.DistributedDatasetSpec


# Load dataset again (you can skip this part if it's already loaded)
dataset, dataset_info = tfds.load(
    'oxford_flowers102',
    as_supervised=True,
    with_info=True
)

train_ds = dataset['train']
val_ds = dataset['validation']
test_ds = dataset['test']

num_train = dataset_info.splits['train'].num_examples
#load label
with open(r'C:\Users\Msys\OneDrive\Documents\vscode\vs1\tensorflow\project2\P2_IMAGE_CLASSIFIER\label_map.json', 'r') as f:
    class_names = json.load(f)

#one example mapping
#print(f"Example mapping (label 1): {class_names['1']}")

# Preprocess(resize + normalize) 

NUM_CLASSES = dataset_info.features['label'].num_classes
NUM_TRAIN = dataset_info.splits['train'].num_examples

IMAGE_RES = 224
BATCH_SIZE = 32
AUTOTUNE = tf.data.AUTOTUNE

def format_image(image, label):
    image = tf.image.resize(image, (IMAGE_RES, IMAGE_RES))
    image = tf.cast(image, tf.float32) / 255.0
    return image, label

train_batches = train_ds.cache() \
                        .shuffle(NUM_TRAIN // 4) \
                        .map(format_image, num_parallel_calls=AUTOTUNE) \
                        .batch(BATCH_SIZE) \
                        .prefetch(AUTOTUNE)

validation_batches = val_ds.cache() \
                            .map(format_image, num_parallel_calls=AUTOTUNE) \
                            .batch(BATCH_SIZE) \
                            .prefetch(AUTOTUNE)

test_batches = test_ds.cache() \
                      .map(format_image, num_parallel_calls=AUTOTUNE) \
                      .batch(BATCH_SIZE) \
                      .prefetch(AUTOTUNE)

 #Build MobileNet classifier



feature_extractor_url = r"C:\Users\Msys\.cache\kagglehub\models\google\mobilenet-v2\tensorFlow2\100-224-feature-vector\2"

# Create the feature extractor layer
feature_extractor_layer = hub.KerasLayer(
    feature_extractor_url,
    input_shape=(IMAGE_RES, IMAGE_RES, 3),
    trainable=False  # freeze base
)

print("Feature extractor loaded successfully!")

# Now build the classifier
model = models.Sequential([
    feature_extractor_layer,
    layers.Dense(NUM_CLASSES, activation='softmax')
])

# Explicitly build the model before summary
model.build((None, IMAGE_RES, IMAGE_RES, 3))
model.summary()


model.compile(
    optimizer='adam',
    loss='sparse_categorical_crossentropy',
    metrics=['accuracy']
)

# 4. Train the model
# -----------------------
EPOCHS = 5

history = model.fit(
    train_batches,
    validation_data=validation_batches,
    epochs=EPOCHS
)

# -----------------------
# 5. Plot training history
# -----------------------
acc = history.history['accuracy']
val_acc = history.history['val_accuracy']
loss = history.history['loss']
val_loss = history.history['val_loss']

epochs_range = range(EPOCHS)

plt.figure(figsize=(12, 5))

# Accuracy plot
plt.subplot(1, 2, 1)
plt.plot(epochs_range, acc, label='Train Accuracy')
plt.plot(epochs_range, val_acc, label='Validation Accuracy')
plt.legend(loc='lower right')
plt.title('Accuracy')

# Loss plot
plt.subplot(1, 2, 2)
plt.plot(epochs_range, loss, label='Train Loss')
plt.plot(epochs_range, val_loss, label='Validation Loss')
plt.legend(loc='upper right')
plt.title('Loss')

plt.show()

# -----------------------
# 6. Evaluate on test set
# -----------------------
test_loss, test_accuracy = model.evaluate(test_batches)
print(f"Test Loss: {test_loss:.4f}")
print(f"Test Accuracy: {test_accuracy:.4f}")

# -----------------------
# 7. Save the trained model
"""model.save("flowers102_mobilenet_model.keras")
print(" Model saved successfully as flowers102_mobilenet_model.keras")"""
import keras
keras.saving.save_model(model, "flowers102_mobilenet_model.keras")


"""model.save("flowers102_mobilenet_model")  # No extension....directory format
print(" Model saved successfully in TensorFlow SavedModel format!")"""


# Load the Keras model
"""loaded_model = load_model(model_path, custom_objects={'KerasLayer': hub.KerasLayer})
print(" Model loaded successfully!")"""

"""model = tf.keras.models.load_model("flowers102_mobilenet_model.keras")
"""

model = load_model("flowers102_mobilenet_model.keras")


# 8. Sanity Check - Top 5 Predictions
# -----------------------
import numpy as np

def process_image(image):
    image = tf.image.resize(image, (IMAGE_RES, IMAGE_RES)) / 255.0
    return image

# Take one example
for image, label in test_ds.take(1):
    plt.imshow(image)
    plt.axis('off')

    img = np.expand_dims(process_image(image), axis=0)
    probs = model.predict(img)[0]
    top_k = 5
    top_indices = probs.argsort()[-top_k:][::-1]
    top_probs = probs[top_indices]
    top_labels = [class_names[str(i+1)] for i in top_indices]

    plt.title("Top Predictions:\n" + "\n".join([f"{top_labels[i]} ({top_probs[i]*100:.1f}%)" for i in range(top_k)]))
    plt.show()

"""#model.save("flowers102_mobilenet_model.h5")
#print("Model saved successfully!")

# -----------------------
# 8. Load the saved model
# -----------------------
loaded_model = load_model(
    "flowers102_mobilenet_model.h5",
    custom_objects={'KerasLayer': hub.KerasLayer}
)
loaded_model.summary()"""

"""prac
def preprocess(image, label):
    image = tf.image.resize(image, (224, 224))
    image = tf.cast(image, tf.float32) / 255.0
    return image, label

    # Convert to Tensor
    image = tf.convert_to_tensor(image, dtype=tf.float32)
    
    # Resize to (224, 224)
    image = tf.image.resize(image, (224, 224))
    
    # Normalize pixel values to 0-1
    image = image / 255.0
    
    # Convert back to NumPy array
    return image.numpy()"""
"""#2
 Prepare one sample from the training dataset
train_ds = train_ds.map(preprocess)
"""
"""BATCH_SIZE = 32
AUTOTUNE = tf.data.AUTOTUNE

train_ds = train_ds.map(preprocess, num_parallel_calls=AUTOTUNE)
train_ds = train_ds.shuffle(1000).batch(BATCH_SIZE).prefetch(AUTOTUNE)

val_ds = val_ds.map(preprocess, num_parallel_calls=AUTOTUNE)
val_ds = val_ds.batch(BATCH_SIZE).prefetch(AUTOTUNE)

test_ds = test_ds.map(preprocess, num_parallel_calls=AUTOTUNE)
test_ds = test_ds.batch(BATCH_SIZE).prefetch(AUTOTUNE)


IMAGE_RES = 224
NUM_CLASSES = dataset_info.features['label'].num_classes  # 102 for flowers102"""

"""# MobileNetV2 feature extractor
feature_extractor_url = "https://tfhub.dev/google/tf2-preview/mobilenet_v2/feature_vector/4"

feature_extractor_layer = hub.KerasLayer(
    feature_extractor_url,
    input_shape=(IMAGE_RES, IMAGE_RES, 3),
    trainable=False  # freeze the convolutional base
)
model = models.Sequential([
    feature_extractor_layer,
    layers.Dense(NUM_CLASSES, activation='softmax')  # output layer
])

model.summary()
model.compile(
    optimizer='adam',
    loss='sparse_categorical_crossentropy',
    metrics=['accuracy']
)
EPOCHS = 5  # you can increase later for better performance

history = model.fit(
    train_batches,
    validation_data=validation_batches,
    epochs=EPOCHS
)
# Extract metrics
acc = history.history['accuracy']
val_acc = history.history['val_accuracy']
loss = history.history['loss']
val_loss = history.history['val_loss']

epochs_range = range(EPOCHS)

plt.figure(figsize=(12, 5))

# Accuracy plot
plt.subplot(1, 2, 1)
plt.plot(epochs_range, acc, label='Train Accuracy')
plt.plot(epochs_range, val_acc, label='Validation Accuracy')
plt.legend(loc='lower right')
plt.title('Accuracy')

# Loss plot
plt.subplot(1, 2, 2)
plt.plot(epochs_range, loss, label='Train Loss')
plt.plot(epochs_range, val_loss, label='Validation Loss')
plt.legend(loc='upper right')
plt.title('Loss')

plt.show()

# Evaluate the model on the test set
test_loss, test_accuracy = model.evaluate(test_batches)

print(f"Test Loss: {test_loss:.4f}")
print(f"Test Accuracy: {test_accuracy:.4f}")"""

""" 4.
# plotting one image with its actual class nam
for image, label in train_ds.take(3):
    plt.imshow(image[0])   # since batching, take first in batch
    label_int = int(label[0].numpy())  # convert Tensor to int
    label_name = class_names[str(label_int + 1)]  # labels start at 1 in JSON
    plt.title(label_name)
    plt.axis('off')
    plt.show()"""
"""1.
num_val = dataset_info.splits['validation'].num_examples
num_test = dataset_info.splits['test'].num_examples
#1
print(f"Training examples: {num_train}")
#2
print(f"Validation examples: {num_val}")
#3
print(f"Testing examples: {num_test}")

# 4 the number of classes
num_classes = dataset_info.features['label'].num_classes
print(f"Number of classes: {num_classes}")

# 5 Print shape and label of 3 images from training set 
for i, (image, label) in enumerate(train_ds.take(3)):
    print(f"Image {i+1} shape: {image.shape}, Label: {label.numpy()}")

# 6 plotting one image
for image, label in train_ds.take(1):
    plt.imshow(image)
    plt.title(f"Label: {label.numpy()}")
    plt.axis("off")
    plt.show()
"""
