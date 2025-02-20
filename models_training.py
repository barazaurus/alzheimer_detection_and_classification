import warnings
import numpy as np
import pandas as pd
import tensorflow as tf
from matplotlib import pyplot as plt
from tensorflow import keras
from tensorflow.keras.applications import InceptionV3, VGG16, ResNet50
from tensorflow.keras.callbacks import CSVLogger, EarlyStopping, ModelCheckpoint
from tensorflow.keras.layers import Dense, GlobalAveragePooling2D, Input, Flatten
from tensorflow.keras.models import Model, load_model

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

    val_ds = tf.keras.preprocessing.image_dataset_from_directory(
        "./output/val",
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

    class_names = train_ds.class_names

    return train_ds, test_ds, val_ds, class_names


def create_custom_model():
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


def create_inception_model():
    # Input layer
    input_tensor = Input(shape=(IMG_HEIGHT, IMG_WIDTH, 3))

    # Load InceptionV3 without top layers
    base_model = InceptionV3(
        weights='imagenet',
        include_top=False,
        input_tensor=input_tensor
    )

    # Freeze the base model layers
    base_model.trainable = False

    # Add custom classification layers
    x = base_model.output
    x = GlobalAveragePooling2D()(x)
    x = Dense(128, activation='relu')(x)
    x = Dense(64, activation='relu')(x)
    predictions = Dense(4, activation='softmax')(x)

    # Create the model
    model = Model(inputs=input_tensor, outputs=predictions)

    # Compile the model
    model.compile(
        optimizer='adam',
        loss='sparse_categorical_crossentropy',
        metrics=['accuracy']
    )

    return model


def create_vgg16_model():
    # Input layer
    input_tensor = Input(shape=(IMG_HEIGHT, IMG_WIDTH, 3))

    # Load VGG16 without top layers
    base_model = VGG16(
        weights='imagenet',
        include_top=False,
        input_tensor=input_tensor
    )

    # Freeze the base model layers
    base_model.trainable = False

    # Add custom classification layers
    x = base_model.output
    x = GlobalAveragePooling2D()(x)
    x = Dense(128, activation='relu')(x)
    x = Dense(64, activation='relu')(x)
    predictions = Dense(4, activation='softmax')(x)

    # Create the model
    model = Model(inputs=input_tensor, outputs=predictions)

    # Compile the model
    model.compile(
        optimizer='adam',
        loss='sparse_categorical_crossentropy',
        metrics=['accuracy']
    )

    return model


def create_resnet_model():
    # Input layer
    input_tensor = Input(shape=(IMG_HEIGHT, IMG_WIDTH, 3))

    # Load ResNet50 without top layers
    base_model = ResNet50(
        weights='imagenet',
        include_top=False,
        input_tensor=input_tensor
    )

    # Freeze the base model layers
    base_model.trainable = False

    # Add custom classification layers
    x = base_model.output
    x = GlobalAveragePooling2D()(x)
    x = Dense(128, activation='relu')(x)
    x = Dense(64, activation='relu')(x)
    predictions = Dense(4, activation='softmax')(x)

    # Create the model
    model = Model(inputs=input_tensor, outputs=predictions)

    # Compile the model
    model.compile(
        optimizer='adam',
        loss='sparse_categorical_crossentropy',
        metrics=['accuracy']
    )

    return model


def train_model(model, train_ds, val_ds, model_name):
    print(f"\nTraining {model_name}...")

    # Define callbacks
    early_stopping = EarlyStopping(
        monitor='val_loss',
        patience=10,
        restore_best_weights=True,
        mode='min',
        verbose=1
    )

    # Add ModelCheckpoint to save the best model during training
    checkpoint = ModelCheckpoint(
        f'{model_name.lower().replace(" ", "_")}.keras',
        monitor='val_accuracy',
        mode='max',
        save_best_only=True,
        verbose=1
    )

    # Train the model with callbacks
    hist = model.fit(
        train_ds,
        validation_data=val_ds,
        epochs=100,
        batch_size=64,
        verbose=1,
        callbacks=[early_stopping, checkpoint]
    )
    return hist


def plot_comparison_results(histories, model_names):
    metrics = ['accuracy', 'loss']
    fig, axes = plt.subplots(1, 2, figsize=(15, 5))

    for metric, ax in zip(metrics, axes):
        for hist, name in zip(histories, model_names):
            ax.plot(hist.history[metric], label=f'{name} (train)')
            ax.plot(hist.history[f'val_{metric}'], label=f'{name} (val)')

        ax.set_title(f'Model {metric.capitalize()}')
        ax.set_xlabel('Epoch')
        ax.set_ylabel(metric.capitalize())
        ax.legend()

    plt.tight_layout()
    plt.show()


def evaluate_models(models, test_ds, model_names):
    results = []
    for model, name in zip(models, model_names):
        print(f"\nEvaluating {name}...")
        test_loss, test_accuracy = model.evaluate(test_ds)
        results.append({
            'Model': name,
            'Test Loss': test_loss,
            'Test Accuracy': test_accuracy
        })

    return pd.DataFrame(results)


if __name__ == '__main__':
    # Load datasets
    train_ds, test_ds, val_ds, class_names = load_datasets()

    # Create models
    custom_model = create_custom_model()
    inception_model = create_inception_model()
    vgg16_model = create_vgg16_model()
    resnet_model = create_resnet_model()

    models = [custom_model, inception_model, vgg16_model, resnet_model]
    model_names = ['Custom CNN', 'InceptionV3', 'VGG16', 'ResNet50']

    # Train models and collect histories
    histories = []
    for model, name in zip(models, model_names):
        history = train_model(model, train_ds, val_ds, name)
        histories.append(history)

        # Save the trained model
        model.save(f'{name.lower().replace(" ", "_")}_model.keras')

    # Plot comparison results
    plot_comparison_results(histories, model_names)

    # Evaluate models on test set
    results_df = evaluate_models(models, test_ds, model_names)


    print("\nModel Evaluation Results:")
    print(results_df)


