import os

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"
import tensorflow as tf
import math
import tensorflow_hub as hub
import numpy as np
import matplotlib.pyplot as plt
from tensorflow import keras
from tensorflow.keras import layers
from sklearn.metrics import roc_curve
from tensorflow.keras.preprocessing.image import ImageDataGenerator

train_examples = 924
validation_examples = 120
test_examples = 124
img_height = img_width = 224 #Det er en standard når man bruger keras, at sætte billed width og height til 256, men det virker ikke. Så vi satte den til 224 i stedet lmao
batch_size = 32

# NasNet - vi skal nok prøve at bruge en anden model, nas er ikke så god
model = keras.Sequential([
    hub.KerasLayer("https://tfhub.dev/google/imagenet/nasnet_mobile/feature_vector/4",
                   trainable=True),
    layers.Dense(1, activation="sigmoid"),
 ])

#model = keras.models.load_model("isic_model/")

#Datagenerators
train_datagen = ImageDataGenerator(
    rescale=1.0 / 255,
    rotation_range=15, #kan evt. være større - læs på data augmentation
    zoom_range=(0.95, 0.95), #zoomer ind og ud tilfældigt med 5%
    horizontal_flip=True,
    vertical_flip=True,
    data_format="channels_last",
    #validation_split (float mellem 0-1)? hvis vi bruger det her bliver alt den ovenstående data augmentation også sat på valisation sættet. fordi validation og test sættet skal ligne hinanden
    dtype=tf.float32,
)

#De næste to linjer skal være ens
validation_datagen = ImageDataGenerator(rescale=1.0 / 255, dtype=tf.float32)
test_datagen = ImageDataGenerator(rescale=1.0 / 255, dtype=tf.float32)

train_gen = train_datagen.flow_from_directory(
    "data3/train/",
    target_size=(img_height, img_width),
    batch_size=batch_size,
    color_mode="rgb",
    class_mode="binary",
    shuffle=True,
    seed=123,
)

validation_gen = validation_datagen.flow_from_directory(
    "data3/validation/",
    target_size=(img_height, img_width),
    batch_size=batch_size,
    color_mode="rgb",
    class_mode="binary",
    shuffle=True,
    seed=123,
)

test_gen = test_datagen.flow_from_directory(
    "data3/test/",
    target_size=(img_height, img_width),
    batch_size=batch_size,
    color_mode="rgb",
    class_mode="binary",
    shuffle=True,
    seed=123,
)

METRICS = [
    keras.metrics.BinaryAccuracy(name="accuracy"),
    keras.metrics.Precision(name="precision"),
    keras.metrics.Recall(name="recall"),
    keras.metrics.AUC(name="auc"),
]

model.compile(
    optimizer=keras.optimizers.Adam(lr=3e-4),
    loss=[keras.losses.BinaryCrossentropy(from_logits=False)],
    metrics=METRICS,
)

model.fit(
    train_gen,
    epochs=10,
    verbose=1,
    steps_per_epoch=train_examples // batch_size,
    validation_data=validation_gen,
    validation_steps=validation_examples // batch_size,
    #Næste linje kode er hvordan man loader en gemt model
    #callbacks=[keras.callbacks.ModelCheckpoint("isic_model")],
)

#Laver en model der viser false positive og true positive
def plot_roc(labels, data):
    predictions = model.predict(data)
    fp, tp, _ = roc_curve(labels, predictions)

    plt.plot(100 * fp, 100 * tp)
    plt.xlabel("False positives [%]")
    plt.ylabel("True positives [%]")
    plt.show()


test_labels = np.array([])
num_batches = 0

#Itererer gennem test generatoren
for _, y in test_gen:
    test_labels = np.append(test_labels, y)
    num_batches += 1
    if num_batches == math.ceil(test_examples / batch_size):
        break

model.evaluate(validation_gen, verbose=1)
model.evaluate(test_gen, verbose=1)
plot_roc(test_labels, test_gen)