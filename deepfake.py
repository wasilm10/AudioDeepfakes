import os
import librosa
import numpy as np
from tqdm import tqdm
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout
from tensorflow.keras.utils import to_categorical

DATASET_PATH = "/Users/wasiahmad/Desktop/English"
TAMPERED_PATH = os.path.join(DATASET_PATH, "tampered")
UNTAMPERED_PATH = os.path.join(DATASET_PATH, "untampered")
SAMPLE_RATE = 16000
DURATION = 3  # seconds
N_MELS = 128
MAX_LEN = 128
SAMPLES_PER_TRACK = SAMPLE_RATE * DURATION

def extract_features(file_path):
    y, sr = librosa.load(file_path, sr=SAMPLE_RATE)
    if len(y) < SAMPLES_PER_TRACK:
        y = np.pad(y, (0, SAMPLES_PER_TRACK - len(y)))
    else:
        y = y[:SAMPLES_PER_TRACK]
    mel = librosa.feature.melspectrogram(y=y, sr=sr, n_mels=N_MELS)
    mel_db = librosa.power_to_db(mel, ref=np.max)
    if mel_db.shape[1] < MAX_LEN:
        mel_db = np.pad(mel_db, ((0, 0), (0, MAX_LEN - mel_db.shape[1])), mode='constant')
    else:
        mel_db = mel_db[:, :MAX_LEN]
    return mel_db

X, y = [], []

print("Loading untampered files...")
for file in tqdm(os.listdir(UNTAMPERED_PATH)):
    if file.endswith(".wav"):
        path = os.path.join(UNTAMPERED_PATH, file)
        X.append(extract_features(path))
        y.append(0)


print("Loading tampered files...")
for subfolder in os.listdir(TAMPERED_PATH):
    sub_path = os.path.join(TAMPERED_PATH, subfolder)
    if os.path.isdir(sub_path):
        for file in os.listdir(sub_path):
            if file.endswith(".wav"):
                path = os.path.join(sub_path, file)
                X.append(extract_features(path))
                y.append(1)

X = np.array(X)[..., np.newaxis]  # Add channel dimension
y = to_categorical(y, num_classes=2)
X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, stratify=y, random_state=42)


model = Sequential([
    Conv2D(32, (3,3), activation='relu', input_shape=X.shape[1:]),
    MaxPooling2D(2,2),
    Conv2D(64, (3,3), activation='relu'),
    MaxPooling2D(2,2),
    Flatten(),
    Dense(128, activation='relu'),
    Dropout(0.3),
    Dense(2, activation='softmax')
])

model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
model.summary()


history = model.fit(X_train, y_train, epochs=15, validation_data=(X_val, y_val), batch_size=16)


y_pred = np.argmax(model.predict(X_val), axis=1)
y_true = np.argmax(y_val, axis=1)
print("\nClassification Report:\n", classification_report(y_true, y_pred))


model.save("deepfake_audio_cnn.h5")
