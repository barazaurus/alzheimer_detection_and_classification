import warnings
import numpy as np
import pandas as pd
import tensorflow as tf
from matplotlib import pyplot as plt
from tensorflow import keras
from tensorflow.keras.callbacks import CSVLogger
from tensorflow.python.keras.layers import Dense
from tensorflow.python.keras.models import load_model



warnings.filterwarnings('ignore')

IMG_HEIGHT = 128
IMG_WIDTH = 128

def load_datasets():
    train_ds = tf.keras.preprocessing.image_dataset_from_directory(
        "./output/train",
        seed=123,
        image_size=(IMG_HEIGHT, IMG_WIDTH),
        batch_size=64
    )

    test_ds = tf.keras.preprocessing.image_dataset_from_directory(
        "./output/test",
        seed=123,
        image_size=(IMG_HEIGHT, IMG_WIDTH),
        batch_size=64
    )

    val_ds = tf.keras.preprocessing.image_dataset_from_directory(
        "./output/val",
        seed=123,
        image_size=(IMG_HEIGHT, IMG_WIDTH),
        batch_size=64
    )
    class_names = train_ds.class_names

    return train_ds, test_ds, val_ds, class_names


def view_data_images(train_ds, class_names):
    plt.figure(figsize=(10, 10))
    for images, labels in train_ds.take(1):
        for i in range(9):
            ax = plt.subplot(3, 3, i + 1)
            plt.imshow(images[i].numpy().astype("uint8"))
            plt.title(class_names[labels[i]])
            plt.axis("off")
    plt.show()


def view_size_diagram(class_names):
    size = [896, 64, 3200, 2240]
    plt.bar(class_names, size)
    plt.xlabel('Class Names')
    plt.ylabel('Size')
    plt.show()


def create_model():
    model = tf.keras.models.Sequential()
    model.add(tf.keras.layers.Rescaling(1. / 255, input_shape=(IMG_HEIGHT, IMG_WIDTH, 3)))
    model.add(tf.keras.layers.Conv2D(filters=16, kernel_size=(3, 3), padding='same', activation='relu',
                                     kernel_initializer="he_normal"))
    model.add(tf.keras.layers.MaxPooling2D(pool_size=(2, 2)))

    model.add(tf.keras.layers.Conv2D(filters=32, kernel_size=(3, 3), padding='same', activation='relu',
                                     kernel_initializer="he_normal"))
    model.add(tf.keras.layers.MaxPooling2D(pool_size=(2, 2)))

    model.add(tf.keras.layers.Dropout(0.20))

    model.add(tf.keras.layers.Conv2D(filters=64, kernel_size=(3, 3), padding='same', activation='relu',
                                     kernel_initializer="he_normal"))
    model.add(tf.keras.layers.MaxPooling2D(pool_size=(2, 2)))

    model.add(tf.keras.layers.Dropout(0.25))
    model.add(tf.keras.layers.Flatten())
    model.add(tf.keras.layers.Dense(128, activation="relu", kernel_initializer="he_normal"))
    model.add(tf.keras.layers.Dense(64, "relu"))
    model.add(tf.keras.layers.Dense(4, "softmax"))

    model.compile(loss="sparse_categorical_crossentropy",
                  optimizer="adam", metrics=["accuracy"])
    return model


def train_model(model, train_ds, val_ds):
    hist = model.fit(train_ds, validation_data=val_ds, epochs=100, batch_size=64, verbose=1)
    return hist


def plot_training_results(hist):
    get_ac = hist.history['accuracy']
    get_los = hist.history['loss']
    val_acc = hist.history['val_accuracy']
    val_loss = hist.history['val_loss']

    epochs = range(len(get_ac))
    plt.figure()
    plt.plot(epochs, get_ac, 'g', label='Accuracy of Training data')
    plt.plot(epochs, get_los, 'r', label='Loss of Training data')
    plt.title('Training data accuracy and loss')
    plt.legend(loc=0)

    plt.figure()
    plt.plot(epochs, get_ac, 'g', label='Accuracy of Training Data')
    plt.plot(epochs, val_acc, 'r', label='Accuracy of Validation Data')
    plt.title('Training and Validation Accuracy')
    plt.legend(loc=0)

    plt.figure()
    plt.plot(epochs, get_los, 'g', label='Loss of Training Data')
    plt.plot(epochs, val_loss, 'r', label='Loss of Validation Data')
    plt.title('Training and Validation Loss')
    plt.legend(loc=0)
    plt.show()


def test_model():
    # Shuffle the test dataset
    shuffled_test_ds = test_ds.shuffle(buffer_size=1000, seed=None)  # None means random seed

    test_images = None
    test_labels = None

    # Take first batch after shuffling
    for images, labels in shuffled_test_ds.take(1):
        test_images = images
        test_labels = labels
        break

    plt.figure(figsize=(20, 20))
    for i in range(16):
        if i < len(test_images):
            ax = plt.subplot(4, 4, i + 1)
            plt.imshow(test_images[i].numpy().astype("uint8"))
            predictions = model.predict(tf.expand_dims(test_images[i], 0))
            score = tf.nn.softmax(predictions[0])

            if class_names[test_labels[i]] == class_names[np.argmax(score)]:
                plt.title("Actual: " + class_names[test_labels[i]])
                plt.ylabel("Predicted: " + class_names[np.argmax(score)],
                           fontdict={'color': 'green', 'fontsize': 6})
            else:
                plt.title("Actual: " + class_names[test_labels[i]])
                plt.ylabel("Predicted: " + class_names[np.argmax(score)],
                           fontdict={'color': 'red', 'fontsize': 6})
            plt.gca().axes.yaxis.set_ticklabels([])
            plt.gca().axes.xaxis.set_ticklabels([])

    plt.subplots_adjust(wspace=0.3, hspace=0.3)
    plt.show()


if __name__ == '__main__':
    train_ds, test_ds, val_ds, class_names = load_datasets()
    #view_data_images(train_ds, class_names)
    #view_size_diagram(class_names)
    #model = create_model()
    #hist = train_model(model, train_ds, val_ds)
    #model.summary()
    #model.save('./saved models/custom_cnn.keras')
    model = tf.keras.models.load_model('./saved models/custom_cnn.keras')
    test_model()
