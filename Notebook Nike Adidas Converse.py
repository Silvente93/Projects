# -*- coding: utf-8 -*-
"""
Created on Wed Sep  7 10:02:12 2022

@author: GUILLE
"""

from keras.layers import Dense, Conv2D, Dropout, Flatten, MaxPooling2D, SeparableConv2D
from keras.callbacks import EarlyStopping
from keras.utils.vis_utils import plot_model
from keras.models import Sequential, save_model, load_model
from keras.preprocessing.image import ImageDataGenerator


classifier = Sequential()

classifier.add(SeparableConv2D(64, (3,3), input_shape=(240,240,3), activation="relu"))
classifier.add(MaxPooling2D(2,2))
classifier.add(SeparableConv2D(128, (3,3), activation="relu"))
classifier.add(MaxPooling2D(2,2))
classifier.add(SeparableConv2D(128, (3,3), activation="relu"))
classifier.add(MaxPooling2D(2,2))

classifier.add(Flatten())

classifier.add(Dense(64, activation="relu", kernel_initializer="uniform"))
classifier.add(Dropout(0.4))
classifier.add(Dense(3, activation="softmax"))

classifier.compile(optimizer="adam", loss="categorical_crossentropy", metrics = ['accuracy'])

classifier.summary()

plot_model(classifier, "classifier.jpg", True)

train_datagen = ImageDataGenerator(
        rescale=1./255,
        shear_range=0.2,
        zoom_range=0.2,
        horizontal_flip=True,
        rotation_range=10
        )

test_datagen = ImageDataGenerator(rescale=1./255)

training_dataset = train_datagen.flow_from_directory("Kaggle_Nike_Adidas_Converse/dataset/train",
                                                     target_size=(240,240),
                                                     batch_size=32)

testing_dataset = train_datagen.flow_from_directory("Kaggle_Nike_Adidas_Converse/dataset/test",
                                                     target_size=(240,240),
                                                     batch_size=32)


callback = EarlyStopping(monitor='loss', patience=3)

history = classifier.fit(training_dataset,
                         epochs=20,
                         validation_data=testing_dataset,
                         workers=3,
                         use_multiprocessing=False,
                         callbacks=[callback])
