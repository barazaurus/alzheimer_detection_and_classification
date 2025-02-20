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
from sklearn.metrics import confusion_matrix, precision_score, recall_score, f1_score
import seaborn as sns

warnings.filterwarnings('ignore')

IMG_HEIGHT = 128
IMG_WIDTH = 128


def plot_confusion_matrix(true_labels, predictions, class_names, title):
    cm = confusion_matrix(true_labels, predictions)
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=class_names, yticklabels=class_names)
    plt.title(f'Confusion Matrix - {title}')
    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')
    plt.tight_layout()
    plt.show()


def calculate_performance_metrics(model_path, test_ds, class_names, model_name):
    try:
        model = tf.keras.models.load_model(model_path)
        predictions = []
        true_labels = []

        for images, labels in test_ds:
            pred = model.predict(images, verbose=0)
            predictions.extend(np.argmax(pred, axis=1))
            true_labels.extend(labels.numpy())

        # Plot confusion matrix for current model
        plot_confusion_matrix(true_labels, predictions, class_names, model_name)

        precision = precision_score(true_labels, predictions, average=None)
        recall = recall_score(true_labels, predictions, average=None)
        f1 = f1_score(true_labels, predictions, average=None)

        metrics_df = pd.DataFrame({
            'Class': class_names,
            'Precision': [f'{x:.3f}' for x in precision],
            'Recall': [f'{x:.3f}' for x in recall],
            'F1-Score': [f'{x:.3f}' for x in f1]
        })

        return metrics_df

    except Exception as e:
        print(f"Error processing model {model_name}: {e}")
        return None


def plot_all_models_metrics(model_paths, class_names):
    plt.figure(figsize=(15, 10))

    all_metrics = []
    for name, path in model_paths.items():
        test_ds = tf.keras.preprocessing.image_dataset_from_directory(
            "./output/test",
            seed=123,
            image_size=(IMG_HEIGHT, IMG_WIDTH),
            batch_size=64
        )

        metrics_df = calculate_performance_metrics(path, test_ds, class_names, name)
        if metrics_df is not None:
            metrics_df['Model'] = name
            all_metrics.append(metrics_df)

    # Combine all metrics
    if all_metrics:
        combined_metrics = pd.concat(all_metrics, axis=0)

        plt.figure(figsize=(15, 10))
        plt.axis('tight')
        plt.axis('off')
        table = plt.table(cellText=combined_metrics.values,
                          colLabels=combined_metrics.columns,
                          cellLoc='center',
                          loc='center',
                          colWidths=[0.2, 0.15, 0.15, 0.15, 0.15])

        table.auto_set_font_size(False)
        table.set_fontsize(9)
        table.scale(1.5, 1.8)

        plt.title('Performance Metrics for All Models', pad=20, size=14)
        plt.show()
    else:
        print("No metrics available to display")


# Evaluate all models
class_names = ['Mild_Demented', 'Moderate_Demented',
               'Non_Demented', 'Very_Mild_Demented']

model_paths = {
    'Custom CNN': './saved models/custom_cnn.keras',
    'VGG16': './saved models/vgg16.keras',
    'InceptionV3': './saved models/inceptionv3.keras',
    'ResNet50': './saved models/resnet50.keras'
}

plot_all_models_metrics(model_paths, class_names)