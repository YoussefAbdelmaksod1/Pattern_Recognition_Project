import cv2
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint, TensorBoard
import os
import time
from sklearn.metrics import classification_report, confusion_matrix
import seaborn as sns
import matplotlib.pyplot as plt
from datetime import datetime
import tkinter as tk
from tkinter import filedialog
from PIL import Image, ImageTk

# ===============================================
# GLOBAL CONFIGURATIONS
# ===============================================
MODEL_PATH = 'emotion_detection_model.h5'
LOG_DIR = "logs/fit/" + datetime.now().strftime("%Y%m%d-%H%M%S")
BATCH_SIZE = 32
IMAGE_SIZE = (48, 48)
emotion_labels = ['Angry', 'Disgust', 'Fear', 'Happy', 'Sad', 'Surprise', 'Neutral']


# ===============================================
# DATA PREPARATION AND MODEL TRAINING
# ===============================================
def load_data():
    data_gen = ImageDataGenerator(
        rescale=1. / 255,
        validation_split=0.2,
        rotation_range=20,
        width_shift_range=0.2,
        height_shift_range=0.2,
        horizontal_flip=True
    )

    train_data = data_gen.flow_from_directory(
        'fer2013_dataset',
        target_size=IMAGE_SIZE,
        color_mode='grayscale',
        class_mode='categorical',
        batch_size=BATCH_SIZE,
        subset='training'
    )

    val_data = data_gen.flow_from_directory(
        'fer2013_dataset',
        target_size=IMAGE_SIZE,
        color_mode='grayscale',
        class_mode='categorical',
        batch_size=BATCH_SIZE,
        subset='validation'
    )
    return train_data, val_data


# ===============================================
# MODEL ARCHITECTURE
# ===============================================
def build_model():
    model = Sequential([
        # First Convolutional Block
        Conv2D(32, (3, 3), activation='relu', input_shape=(48, 48, 1),
               kernel_regularizer=tf.keras.regularizers.l2(0.01)),
        MaxPooling2D(pool_size=(2, 2)),

        # Second Convolutional Block
        Conv2D(64, (3, 3), activation='relu',
               kernel_regularizer=tf.keras.regularizers.l2(0.01)),
        MaxPooling2D(pool_size=(2, 2)),

        # Third Convolutional Block
        Conv2D(128, (3, 3), activation='relu',
               kernel_regularizer=tf.keras.regularizers.l2(0.01)),
        MaxPooling2D(pool_size=(2, 2)),

        # Flatten and Dense Layers
        Flatten(),
        Dense(256, activation='relu'),
        Dropout(0.5),
        Dense(128, activation='relu'),
        Dropout(0.3),
        Dense(7, activation='softmax')
    ])

    return model


# ===============================================
# TRAINING MONITORING
# ===============================================
class TimeHistory(tf.keras.callbacks.Callback):
    def on_train_begin(self, logs={}):
        self.times = []

    def on_epoch_begin(self, epoch, logs={}):
        self.epoch_time_start = time.time()

    def on_epoch_end(self, epoch, logs={}):
        self.times.append(time.time() - self.epoch_time_start)


# ===============================================
# MODEL TRAINING AND EVALUATION
# ===============================================
def train_and_evaluate_model(model, train_data, val_data):
    early_stopping = EarlyStopping(
        monitor='val_loss',
        patience=5,
        restore_best_weights=True
    )

    checkpoint = ModelCheckpoint(
        MODEL_PATH,
        monitor='val_accuracy',
        save_best_only=True
    )

    tensorboard_callback = TensorBoard(
        log_dir=LOG_DIR,
        histogram_freq=1
    )

    time_callback = TimeHistory()

    model.compile(
        optimizer='adam',
        loss='categorical_crossentropy',
        metrics=['accuracy',
                 tf.keras.metrics.Precision(),
                 tf.keras.metrics.Recall()]
    )

    history = model.fit(
        train_data,
        validation_data=val_data,
        epochs=30,
        callbacks=[early_stopping, checkpoint, tensorboard_callback, time_callback]
    )

    return history, time_callback


# ===============================================
# PERFORMANCE EVALUATION
# ===============================================
def evaluate_model_performance(model, val_data, time_callback):
    val_predictions = model.predict(val_data)
    y_pred = np.argmax(val_predictions, axis=1)
    y_true = val_data.classes

    class_report = classification_report(y_true, y_pred, target_names=emotion_labels)
    conf_matrix = confusion_matrix(y_true, y_pred)

    plt.figure(figsize=(10, 8))
    sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues',
                xticklabels=emotion_labels,
                yticklabels=emotion_labels)
    plt.title('Confusion Matrix')
    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')
    plt.savefig('confusion_matrix.png')

    avg_epoch_time = np.mean(time_callback.times)
    print(f"\nAverage time per epoch: {avg_epoch_time:.2f} seconds")

    return class_report, conf_matrix


# ===============================================
# PREPROCESSING FUNCTIONS
# ===============================================
def preprocess_frame(frame):
    try:
        if len(frame.shape) == 3:
            gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        else:
            gray_frame = frame

        resized_frame = cv2.resize(gray_frame, (48, 48))
        normalized_frame = resized_frame / 255.0
        reshaped_frame = np.reshape(normalized_frame, (1, 48, 48, 1))

        return reshaped_frame

    except Exception as e:
        print(f"Error in preprocessing frame: {str(e)}")
        return None


# ===============================================
# INFERENCE OPTIMIZATION
# ===============================================
def optimize_inference(frame, model):
    start_time = time.time()

    frame = tf.cast(frame, tf.float16)

    @tf.function
    def optimized_predict(img):
        return model(img, training=False)

    prediction = optimized_predict(frame)

    inference_time = time.time() - start_time
    return prediction, inference_time


# ===============================================
# REAL-TIME DETECTION
# ===============================================
def real_time_emotion_detection():
    cap = cv2.VideoCapture(0)
    face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

    frame_times = []
    fps_list = []

    print("Starting real-time emotion detection. Press 'q' to quit.")

    while True:
        frame_start_time = time.time()

        ret, frame = cap.read()
        if not ret:
            break

        gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = face_cascade.detectMultiScale(gray_frame, scaleFactor=1.3, minNeighbors=5)

        for (x, y, w, h) in faces:
            face_roi = frame[y:y + h, x:x + w]
            preprocessed_face = preprocess_frame(face_roi)

            if preprocessed_face is not None:
                predictions, inference_time = optimize_inference(preprocessed_face, model)
                emotion_index = np.argmax(predictions)
                emotion = emotion_labels[emotion_index]
                confidence = float(predictions[0][emotion_index])

                cv2.rectangle(frame, (x, y), (x + w, y + h), (255, 0, 0), 2)
                text = f"{emotion}: {confidence:.2f}"
                cv2.putText(frame, text, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 0, 0), 2)
                cv2.putText(frame, f"Inference: {inference_time * 1000:.1f}ms",
                            (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

        frame_time = time.time() - frame_start_time
        fps = 1.0 / frame_time
        fps_list.append(fps)
        cv2.putText(frame, f"FPS: {fps:.1f}", (10, 60),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

        cv2.imshow('Real-Time Emotion Detection', frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    avg_fps = np.mean(fps_list)
    print(f"\nAverage FPS: {avg_fps:.2f}")

    cap.release()
    cv2.destroyAllWindows()


# ===============================================
# STATIC IMAGE PROCESSING
# ===============================================
def process_static_image(image_path):
    try:
        image = cv2.imread(image_path)
        if image is None:
            raise Exception("Could not read the image")

        window_name = "Emotion Detection Result"
        cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)

        gray_frame = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        face_cascade = cv2.CascadeClassifier(
            cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'
        )
        faces = face_cascade.detectMultiScale(gray_frame, scaleFactor=1.3, minNeighbors=5)

        for (x, y, w, h) in faces:
            face_roi = image[y:y + h, x:x + w]
            preprocessed_face = preprocess_frame(face_roi)

            if preprocessed_face is not None:
                predictions, inference_time = optimize_inference(preprocessed_face, model)
                emotion_index = np.argmax(predictions)
                emotion = emotion_labels[emotion_index]
                confidence = float(predictions[0][emotion_index])

                cv2.rectangle(image, (x, y), (x + w, y + h), (255, 0, 0), 2)
                text = f"{emotion}: {confidence:.2f}"
                cv2.putText(image, text, (x, y - 10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 0, 0), 2)
                cv2.putText(image, f"Inference: {inference_time * 1000:.1f}ms",
                            (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

        cv2.imshow(window_name, image)

        result_path = f"result_{os.path.basename(image_path)}"
        cv2.imwrite(result_path, image)
        print(f"Result saved as: {result_path}")

        cv2.waitKey(0)
        cv2.destroyAllWindows()

    except Exception as e:
        print(f"Error processing image: {str(e)}")
        cv2.destroyAllWindows()


# ===============================================
# GUI IMPLEMENTATION
# ===============================================
class EmotionDetectionGUI:
    def __init__(self):
        self.root = tk.Tk()
        self.root.title("Emotion Detection System")
        self.root.geometry("500x300")
        self.setup_gui()

    def setup_gui(self):
        title = tk.Label(self.root,
                         text="Emotion Detection System",
                         font=("Arial", 20, "bold"))
        title.pack(pady=20)

        btn_frame = tk.Frame(self.root)
        btn_frame.pack(pady=30)

        real_time_btn = tk.Button(btn_frame,
                                  text="Real-time Detection",
                                  command=self.start_real_time,
                                  width=20,
                                  height=2)
        real_time_btn.pack(pady=10)

        image_btn = tk.Button(btn_frame,
                              text="Choose Image",
                              command=self.process_image,
                              width=20,
                              height=2)
        image_btn.pack(pady=10)

        quit_btn = tk.Button(btn_frame,
                             text="Quit",
                             command=self.root.quit,
                             width=20,
                             height=2)
        quit_btn.pack(pady=10)

    def start_real_time(self):
        self.root.destroy()
        real_time_emotion_detection()

    def process_image(self):
        file_path = filedialog.askopenfilename(
            title="Choose an image",
            filetypes=[
                ("Image files", "*.png *.jpg *.jpeg *.bmp *.gif"),
                ("All files", "*.*")
            ]
        )
        if file_path:
            self.root.destroy()
            process_static_image(file_path)


# ===============================================
# MAIN EXECUTION
# ===============================================
if __name__ == "__main__":
    # Load or train the model
    if not os.path.exists(MODEL_PATH):
        print("Training new model...")
        train_data, val_data = load_data()
        model = build_model()
        history, time_callback = train_and_evaluate_model(model, train_data, val_data)

        class_report, conf_matrix = evaluate_model_performance(model, val_data, time_callback)
        print("\nClassification Report:")
        print(class_report)

    else:
        print("Loading existing model...")
        model = load_model(MODEL_PATH)

    # Start the GUI
    app = EmotionDetectionGUI()
    app.root.mainloop()