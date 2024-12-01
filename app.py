import os
import streamlit as st
import tensorflow as tf
import numpy as np
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Dense, Flatten, Dropout
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.utils import to_categorical
from sklearn.model_selection import train_test_split
from PIL import Image
import matplotlib.pyplot as plt

# Daftar kelas
CLASS_LABELS = ["Arborio", "Basmati", "Ipsala", "Jasmine", "Karacadag"]

# Dataset Path
DATASET_PATH = "./dataset"  # Sesuaikan lokasi dataset Anda

# Load and preprocess dataset
@st.cache_data
def load_data():
    images = []
    labels = []

    for class_index, class_name in enumerate(CLASS_LABELS):
        class_path = os.path.join(DATASET_PATH, class_name)
        if os.path.isdir(class_path):
            for image_file in os.listdir(class_path):
                image_path = os.path.join(class_path, image_file)
                try:
                    # Load and preprocess images
                    img = tf.keras.preprocessing.image.load_img(image_path, target_size=(224, 224))
                    img_array = tf.keras.preprocessing.image.img_to_array(img) / 255.0
                    images.append(img_array)
                    labels.append(class_index)
                except Exception as e:
                    print(f"Error loading image {image_path}: {e}")

    images = np.array(images)
    labels = np.array(labels)
    return images, labels

# Load dataset
images, labels = load_data()

# Split dataset
labels_categorical = to_categorical(labels)
X_train, X_test, y_train, y_test = train_test_split(images, labels_categorical, test_size=0.2, random_state=42)

# Build CNN model
@st.cache_resource
def build_model():
    model = Sequential([
        Conv2D(32, kernel_size=(3, 3), activation='relu', input_shape=(224, 224, 3)),
        MaxPooling2D(pool_size=(2, 2)),
        Flatten(),
        Dense(128, activation='relu'),
        Dropout(0.3),
        Dense(len(CLASS_LABELS), activation='softmax')
    ])
    model.compile(optimizer=Adam(learning_rate=0.001), loss='categorical_crossentropy', metrics=['accuracy'])
    return model

# Train model
@st.cache_resource
def train_model():
    model = build_model()
    model.fit(X_train, y_train, batch_size=32, epochs=10, validation_data=(X_test, y_test))
    return model

model = train_model()

# Streamlit App
def main():
    st.title("Klasifikasi Citra Padi dengan CNN")
    st.write("Aplikasi ini mengklasifikasikan gambar padi ke dalam 5 kelas: Arborio, Basmati, Ipsala, Jasmine, Karacadag.")

    # Pilih gambar untuk diklasifikasikan
    uploaded_file = st.file_uploader("Unggah gambar padi (jpg, png, jpeg)", type=["jpg", "png", "jpeg"])
    
    if uploaded_file is not None:
        # Tampilkan gambar
        image = Image.open(uploaded_file)
        st.image(image, caption="Gambar yang diunggah", use_column_width=True)

        # Simpan sementara untuk prediksi
        with open("uploaded_image.jpg", "wb") as f:
            f.write(uploaded_file.getbuffer())

        # Prediksi
        if st.button("Klasifikasikan Gambar"):
            img = tf.keras.preprocessing.image.load_img("uploaded_image.jpg", target_size=(224, 224))
            img_array = tf.keras.preprocessing.image.img_to_array(img) / 255.0
            img_array = np.expand_dims(img_array, axis=0)
            
            predictions = model.predict(img_array)
            class_idx = np.argmax(predictions)
            st.write(f"Prediksi kelas gambar: **{CLASS_LABELS[class_idx]}**")
            
            # Hapus file sementara
            os.remove("uploaded_image.jpg")

if __name__ == "__main__":
    main()
