#importing necessary packages
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.layers import AveragePooling2D, Dropout, Flatten, Dense, Input
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input
from tensorflow.keras.preprocessing.image import img_to_array
from tensorflow.keras.preprocessing.image import load_img
from tensorflow.keras.utils import to_categorical
from sklearn.preprocessing import LabelBinarizer
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from imutils import paths
import matplotlib.pyplot as plt
import numpy as np
import os

# Initialising  initial learning rate 
init_lr = 1e-4
epochs = 50
bs = 32

directory = r"D:\dataset"
categories = ["Masked", "Unmasked"]

print('Loading images..')

data =[]
labels =[]

for category in categories:
    path = os.path.join(directory, category)
    for img in os.listdir(path):
        img_path = os.path.join(path, img)
        image = load_img(img_path, target_size=(1024,1024))
        image = img_to_array(image)
        image = preprocess_input(image)

        data.append(image)
        labels.append(category)


 # Converting labels to arrays
lb = LabelBinarizer()
labels = lb.fit_transform(labels)
labels = to_categorical(labels)

data = np.array(data, dtype="float32")
labels = np.array(labels)

(trainX, testX, trainY, testY) = train_test_split(data, labels, test_size = 0.20, stratify=labels, random_state = 42)

# construct the training image generator for data augmentation
aug = ImageDataGenerator(
	rotation_range=20,
	zoom_range=0.15,
	width_shift_range=0.2,
	height_shift_range=0.2,
	shear_range=0.15,
	horizontal_flip=True,
	fill_mode="nearest")



 #Creating base model using MobileNetV2 
baseModel = MobileNetV2(weights="imagenet", include_top= False, input_tensor = Input(shape=(1024,1024,3)))

 # Constructing head model
headModel = baseModel.output
headModel = AveragePooling2D(pool_size=(7, 7))(headModel)
headModel = Flatten(name="flatten")(headModel)
headModel = Dense(128, activation="relu")(headModel)
headModel = Dropout(0.5)(headModel)
headModel = Dense(2, activation="softmax")(headModel)

 #Placing headmodel over baseModel
model = Model(inputs=baseModel.input, outputs = headModel)


for layer in baseModel.layers:
    layer.trainable = False

#Compiling Model
print('Compiling model..')

opt = Adam(lr= init_lr, decay= init_lr/epochs)
model.compile(loss="binary_crossentropy", optimizer=opt, metrics=["accuracy"])



H = model.fit(
    aug.flow(trainX, trainY, batch_size=BS),
	steps_per_epoch=len(trainX) // BS,
	validation_data=(testX, testY),
	validation_steps=len(testX) // BS,
	epochs=epochs
)

#Predictions on dataset
pred = model.predict(textX, batch_size=bs)

pred = np.argmax(pred, axis=1)

print(classification_report(testY.argmax(axis=1), pred,
	target_names=lb.classes_))


model.save("mask_detection.model", save_format="h5")

#Plotting train loss and accuracy
n= epochs
plt.style.use("ggplot")
plt.figure()
plt.plot(np.arange(0, N), H.history["loss"], label="train_loss")
plt.plot(np.arange(0, N), H.history["val_loss"], label="val_loss")
plt.plot(np.arange(0, N), H.history["accuracy"], label="train_acc")
plt.plot(np.arange(0, N), H.history["val_accuracy"], label="val_acc")
plt.title("Training Loss and Accuracy")
plt.xlabel("Epoch #")
plt.ylabel("Loss/Accuracy")
plt.legend(loc="lower left")
plt.savefig("plot.png")

