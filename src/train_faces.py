# Imports
import os
import cv2 as cv
import numpy as np


class TrainFaces:
    def __init__(self, cascade_classifier, face_recognizer, cascade_path, output_path, img_path, labels):
        self.x_train = []
        self.y_labels = []
        self.current_id = 0
        self.cascade_classifier = cascade_classifier
        self.face_recognizer = face_recognizer
        self.cascade_path = cascade_path
        self.output_path = output_path
        self.img_paths = img_path
        self.label_ids = labels

    def run(self):
        for img in self.img_paths:
            # Get folder name (Label)
            label = os.path.basename(os.path.dirname(img)).replace("-", " ").lower()

            # Create IDs for each labels (People)
            if not label in self.label_ids.keys():
                self.label_ids[label] = self.current_id
                self.current_id += 1

            # Get people ID
            id = self.label_ids[label]

            # Read image in grayscale
            image = cv.imread(img, cv.IMREAD_GRAYSCALE)

            # Convert image to numpy array
            image_array = np.array(image, "uint8")

            # Detect faces
            faces = self.cascade_classifier.detectMultiScale(
                image=image_array,
                scaleFactor=1.2,
                minNeighbors=5,
            )

            for (x, y, w, h) in faces:
                roi = image_array[y:y + h, x:x + w]
                self.x_train.append(roi)
                self.y_labels.append(id)

        self.face_recognizer.train(self.x_train, np.array(self.y_labels))
        self.face_recognizer.save(self.output_path)

