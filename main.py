import tensorflow as tf
import numpy as np
import cv2
import glob
import pickle
import random
import os

DATADIR = "D:/dev/set"
CATEGORIES = ["memes", "paisajes", "texto"]
IMG_SIZE = 50

def create_training_data():
    training_data = []
    for category in CATEGORIES:
        path = os.path.join(DATADIR, category)
        class_num = CATEGORIES.index(category)
        for img_path in glob.glob(os.path.join(path, "*.jpg")):
            try:
                img_array = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
                new_array = cv2.resize(img_array, (IMG_SIZE, IMG_SIZE))
                training_data.append([new_array, class_num])
            except Exception as e:
                print(f"Error processing {img_path}: {str(e)}")
    random.shuffle(training_data)
    X = np.array([np.array(x[0]) for x in training_data]).reshape(-1, IMG_SIZE, IMG_SIZE, 1)
    y = np.array([x[1] for x in training_data])
    return X, y

X, y = create_training_data()

X = X / 255.0

pickle_out = open("X.pickle", "wb")
pickle.dump(X, pickle_out)
pickle_out.close()

pickle_out = open("y.pickle", "wb")
pickle.dump(y, pickle_out)
pickle_out.close()

pickle_in = open("X.pickle", "rb")
X = pickle.load(pickle_in)
pickle_in.close()

pickle_in = open("y.pickle", "rb")
y = pickle.load(pickle_in)
pickle_in.close()

model = tf.keras.models.Sequential([
    tf.keras.layers.Conv2D(64, (3, 3), input_shape=X.shape[1:]),
    tf.keras.layers.Activation("relu"),
    tf.keras.layers.MaxPooling2D(pool_size=(2, 2)),
    tf.keras.layers.Conv2D(64, (3, 3)),
    tf.keras.layers.Activation("relu"),
    tf.keras.layers.MaxPooling2D(pool_size=(2, 2)),
    tf.keras.layers.Flatten(),
    tf.keras.layers.Dense(64),
    tf.keras.layers.Activation("relu"),
    tf.keras.layers.Dense(4),
    tf.keras.layers.Activation("softmax")
])
model.compile(loss="sparse_categorical_crossentropy", optimizer="adam", metrics=["accuracy"])
model.fit(X, y, batch_size=32, epochs=10, validation_split=0.1)
model.save("clasificador_imagenes.model")
new_model = tf.keras.models.load_model("clasificador_imagenes.model")

DATADIRTEST = "D:/dev/test" # Path to test images ramdon folder without categories

def prepare(filepath):
    IMG_SIZE = 50
    img_array = cv2.imread(filepath, cv2.IMREAD_GRAYSCALE)
    new_array = cv2.resize(img_array, (IMG_SIZE, IMG_SIZE))
    return new_array.reshape(-1, IMG_SIZE, IMG_SIZE, 1)

for img_path in glob.iglob(os.path.join(DATADIRTEST, "*.jpg")):
    try:
        img_arr = prepare(img_path)
        prediction = new_model.predict(img_arr)
        category = CATEGORIES[int(np.argmax(prediction))]
        os.rename(img_path, os.path.join(DATADIR, category, os.path.basename(img_path)))
        print(f"Image {img_path} moved to {category}")
    except Exception as e:
        print(f"Error processing {img_path}: {str(e)}")


def prepare(filepath):
    img_array = cv2.imread(filepath, cv2.IMREAD_GRAYSCALE)
    new_array = cv2.resize(img_array, (IMG_SIZE, IMG_SIZE))
    return new_array.reshape(-1, IMG_SIZE, IMG_SIZE, 1)