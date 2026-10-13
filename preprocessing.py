
import cv2
import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from pathlib import Path
import tensorflow as tf
import keras


image_folder = Path("Image folder")
categories = ["Table", "Chair", "Bin", "Stabler", "Telephone", "Book", "Pens", "Computer"]

image_data = {}

for category in categories:
    image_data[category] = []
    category_folder = image_folder / category

    for image_file in category_folder.iterdir():
        if image_file.suffix.lower() in [".jpg", ".jpeg", ".png"]:
            image = cv2.imread(str(image_file))

            if image is not None:
                image = cv2.resize(image, (224, 224))
                image_data[category].append(image)

print("Total images loaded:", sum(len(images) for images in image_data.values()))
print("Image sizes:", {category: len(images) for category, images in image_data.items()})

X = []
y = []

for category, images in image_data.items():
    for image in images:
        X.append(image)
        y.append(category)



X = __import__("numpy").array(X)
y = __import__("numpy").array(y)

X_train, X_temp, y_train, y_temp = train_test_split(
    X, y, test_size=0.30, random_state=42, stratify=y
)

X_val, X_test, y_val, y_test = train_test_split(
    X_temp, y_temp, test_size=0.50, random_state=42, stratify=y_temp
)

print("Training images:", len(X_train))
print("Validation images:", len(X_val))
print("Test images:", len(X_test))


class ImagePreprocessor():
    def __init__(self, name):
        self.name = name
        self.images = []

    def add_image(self, image):
        self.images.append(image)

    def blur_background(self):
        pass
        

        
class Bin(ImagePreprocessor):
    pass

class Book(ImagePreprocessor):
    pass

class Chair(ImagePreprocessor):
    pass    

class Computer(ImagePreprocessor):
    pass

class Pen(ImagePreprocessor):
    pass

class Stabler(ImagePreprocessor):
    pass

class Table(ImagePreprocessor):
    pass    

class Telephone(ImagePreprocessor):
    pass










