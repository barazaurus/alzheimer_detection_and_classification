import os
import tkinter as tk
from tkinter import *
from tkinter import filedialog
from PIL import Image, ImageTk
import tensorflow as tf
import numpy as np

IMG_HEIGHT = 128
IMG_WIDTH = 128
class_names = ['Mild_Demented', 'Moderate_Demented', 'Non_Demented', 'Very_Mild_Demented']

MODELS = {
    'Custom CNN': './saved models/custom_cnn.keras',
    'VGG16': './saved models/vgg16.keras',
    'InceptionV3': './saved models/inceptionv3.keras',
    'ResNet50': './saved models/resnet50.keras'
}

# Load all models at startup
loaded_models = {name: tf.keras.models.load_model(path) for name, path in MODELS.items()}


def get_actual_class(file_path):
    filename = os.path.basename(file_path)
    filename = filename.lower()

    if 'mild_' in filename and 'very' not in filename:
        return 'Mild_Demented'
    elif 'moderate_' in filename:
        return 'Moderate_Demented'
    elif 'non_' in filename:
        return 'Non_Demented'
    elif 'verymild_' in filename:
        return 'Very_Mild_Demented'
    return None


def preprocess_image(image_path):
    img = tf.io.read_file(image_path)
    img = tf.image.decode_jpeg(img, channels=3)
    img = tf.image.resize(img, [IMG_HEIGHT, IMG_WIDTH])
    img = tf.expand_dims(img, 0)
    return img


def uploadImage():
    file_path = filedialog.askopenfilename(
        initialdir=os.getcwd(),
        title='Select Image File',
        filetypes=(("JPG File", "*.jpg"), ("PNG file", "*.png"), ("All Files", "*.*"))
    )

    if not file_path:
        return

    # Display original image
    sig_image = Image.open(file_path)
    sig_image.thumbnail((200, 200))
    sig_image = ImageTk.PhotoImage(sig_image)
    image_label.configure(image=sig_image, width=200, height=200)
    image_label.image = sig_image

    # Get actual class
    actual_class = get_actual_class(file_path)
    actual_text.configure(text=f"Actual: {actual_class}")

    # Process image once
    processed_image = preprocess_image(file_path)

    # Clear previous results
    for widget in results_frame.winfo_children():
        if widget != actual_text:
            widget.destroy()

    # Make predictions with all models
    for model_name, model in loaded_models.items():
        prediction = model.predict(processed_image, verbose=0)
        predicted_class = class_names[np.argmax(prediction[0])]

        result_text = Label(
            results_frame,
            text=f"{model_name}: {predicted_class}",
            font=("Helvetica", 16),
            background='#B8DBD9',
            fg="green" if predicted_class == actual_class else "red"
        )
        result_text.pack(pady=5)


class CustomButton(tk.Canvas):
    def __init__(self, parent, width, height, corner_radius, command, color, text='Choose MRI', hover_color=None):
        super().__init__(parent, width=width, height=height, highlightthickness=0, bg='#B8DBD9', relief='ridge')
        self.command = command
        self.color = color
        self.hover_color = hover_color if hover_color else self._darker(color, 20)
        self.corner_radius = corner_radius
        self.text = text

        self.rect = self.create_rounded_rect(0, 0, width, height, corner_radius, fill=color)
        self.text_item = self.create_text(width / 2, height / 2, text=text,
                                          fill='#DFDEDF', font=('Helvetica', 14, 'bold'))

        self.bind('<Enter>', self._on_enter)
        self.bind('<Leave>', self._on_leave)
        self.bind('<Button-1>', self._on_click)
        self.bind('<ButtonRelease-1>', self._on_release)

    def create_rounded_rect(self, x1, y1, x2, y2, radius, **kwargs):
        points = [
            x1 + radius, y1,
            x2 - radius, y1,
            x2, y1,
            x2, y1 + radius,
            x2, y2 - radius,
            x2, y2,
            x2 - radius, y2,
            x1 + radius, y2,
            x1, y2,
            x1, y2 - radius,
            x1, y1 + radius,
            x1, y1
        ]
        return self.create_polygon(points, smooth=True, **kwargs)

    def _darker(self, color, factor=20):
        # Convert hex to RGB
        r = int(color[1:3], 16)
        g = int(color[3:5], 16)
        b = int(color[5:7], 16)

        # Make darker
        r = max(0, r - factor)
        g = max(0, g - factor)
        b = max(0, b - factor)

        return f'#{r:02x}{g:02x}{b:02x}'

    def _on_enter(self, event):
        self.itemconfig(self.rect, fill=self.hover_color)
        self.configure(cursor='hand2')

    def _on_leave(self, event):
        self.itemconfig(self.rect, fill=self.color)
        self.configure(cursor='')

    def _on_click(self, event):
        self.itemconfig(self.rect, fill=self._darker(self.color, 30))

    def _on_release(self, event):
        self.itemconfig(self.rect, fill=self.hover_color)
        self.command()


def create_gui():
    window = tk.Tk()
    window.iconbitmap('./assets/favicon.ico')
    window.title("Alzheimer Detection and Classification")
    window['background'] = '#B8DBD9'
    window_width = 600
    window_height = 700  # Increased height for multiple results

    screen_width = window.winfo_screenwidth()
    screen_height = window.winfo_screenheight()
    center_x = int(screen_width / 2 - window_width / 2)
    center_y = int(screen_height / 2 - window_height / 2)
    window.geometry(f'{window_width}x{window_height}+{center_x}+{center_y}')
    window.resizable(False, False)

    global image_label, actual_text, results_frame

    title_label = Label(
        window,
        text="Alzheimer's MRI Classification",
        font=("Helvetica", 24, "bold"),
        background='#B8DBD9'
    )
    title_label.pack(pady=20)

    image_label = Label(window, background='#B8DBD9')
    image_label.pack(padx=10, pady=10)

    results_frame = Frame(window, background='#B8DBD9')
    results_frame.pack(pady=10)

    actual_text = Label(
        results_frame,
        font=("Helvetica", 16),
        background='#B8DBD9'
    )
    actual_text.pack(pady=5)

    button_frame = Frame(window, background='#B8DBD9', height=100)
    button_frame.pack(side=BOTTOM, fill=X, pady=20)
    button_frame.pack_propagate(False)

    button = CustomButton(
        button_frame,
        width=200,
        height=50,
        corner_radius=15,
        command=uploadImage,
        color='#6E6D70',
        text='Choose MRI'
    )
    button.pack(expand=True)

    return window


if __name__ == '__main__':
    window = create_gui()
    window.mainloop()