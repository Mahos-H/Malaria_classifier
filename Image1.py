import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
import os
from tensorflow.keras import layers
from tensorflow.keras.callbacks import EarlyStopping
from datetime import datetime

timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
test_dir = "./cell_images/test"
data_dir = "./cell_images/cell_images"

batch_size = 32
img_height = 100
img_width = 100

train = tf.keras.utils.image_dataset_from_directory(
    data_dir,
    validation_split=0.2,
    subset="training",
    seed=123,
    image_size=(img_height, img_width),
    batch_size=batch_size
)

valid = tf.keras.utils.image_dataset_from_directory(
    data_dir,
    validation_split=0.2,
    subset="validation",
    seed=123,
    image_size=(img_height, img_width),
    batch_size=batch_size
)

augmentation = tf.keras.Sequential([
    layers.RandomFlip("horizontal", input_shape=(img_height, img_width, 3)),
    layers.RandomRotation(0.1),
    layers.RandomZoom(0.1),
])

classes = train.class_names

AUTOTUNE = tf.data.AUTOTUNE
train = train.cache().shuffle(1000).prefetch(buffer_size=AUTOTUNE)
valid = valid.cache().prefetch(buffer_size=AUTOTUNE)

train = train.map(lambda x, y: (augmentation(x), y))

normalization = layers.Rescaling(1./255)
train = train.map(lambda x, y: (normalization(x), y))
valid = valid.map(lambda x, y: (normalization(x), y))

num_classes = len(classes)
model = tf.keras.Sequential([
    layers.Conv2D(16, 3, padding='same', activation='relu', input_shape=(img_height, img_width, 3)),
    layers.MaxPooling2D(),
    layers.Conv2D(32, 3, padding='same', activation='relu'),
    layers.MaxPooling2D(),
    layers.Conv2D(64, 3, padding='same', activation='relu'),
    layers.MaxPooling2D(),
    layers.Flatten(),
    layers.Dense(128, activation='relu'),
    layers.Dense(num_classes)
])

model.compile(optimizer='adam',
              loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
              metrics=['accuracy'])
model.summary()

early_stop = EarlyStopping(
    monitor='val_loss',
    patience=4,
    restore_best_weights=True
)

epochs = 2
history = model.fit(
    train,
    validation_data=valid,
    epochs=epochs,
    callbacks=[early_stop]
)

acc = history.history['accuracy']
val_acc = history.history['val_accuracy']
loss = history.history['loss']
val_loss = history.history['val_loss']
epochs_range = range(len(acc)) 

plt.figure(figsize=(8, 8))
plt.subplot(1, 2, 1)
plt.plot(epochs_range, acc, label='Training Accuracy')
plt.plot(epochs_range, val_acc, label='Validation Accuracy')
plt.legend(loc='lower right')
plt.title('Training and Validation Accuracy')

plt.subplot(1, 2, 2)
plt.plot(epochs_range, loss, label='Training Loss')
plt.plot(epochs_range, val_loss, label='Validation Loss')
plt.legend(loc='upper right')
plt.title('Training and Validation Loss')
plt.show()

test = tf.keras.utils.image_dataset_from_directory(
    test_dir,
    labels='inferred',
    image_size=(img_height, img_width),
    batch_size=batch_size
)

test = test.map(lambda x, y: (normalization(x), y))

correct_preds = 0
incorrect_preds = 0

for images, labels in test:
    predictions = model.predict(images)
    predicted_classes = np.argmax(predictions, axis=1)
    for pred, true in zip(predictions, labels):
        pred_class = classes[np.argmax(pred)]
        true_class = classes[true]
        if pred_class == true_class and np.max(pred) >= 0.5:
            correct_preds += 1
        else:
            incorrect_preds += 1

plt.figure(figsize=(8, 5))
plt.bar(['Correct Predictions', 'Incorrect Predictions'], [correct_preds, incorrect_preds], color=['blue', 'red'])
plt.xlabel('Prediction Type')
plt.ylabel('Number of Predictions')
plt.title('Number of Correct and Incorrect Predictions')
plt.show()

model_dir = './cell_images'
model_name = f"model_{timestamp}.keras"
os.makedirs(model_dir, exist_ok=True)
model_path = os.path.join(model_dir, model_name)

model.save(model_path)
print("Model saved successfully.")
