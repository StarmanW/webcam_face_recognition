from src.recognize import FaceRecognition
from src.train_faces import TrainFaces
from src.files_utility import FilesUtility
import cv2 as cv

# CONSTANT
CASCADE_PATH = '../cascade/data/haarcascade_frontalface_default.xml'
RESULT_OUTPUT_PATH = '../results/trained_result.yml'

# Variables declaration
labels = {}
img_paths = FilesUtility.getImages('../img')
face_cascade = cv.CascadeClassifier(CASCADE_PATH)

# Create a LBP face recognizer
recognizer = cv.face.LBPHFaceRecognizer_create()

# Train faces
train_faces = TrainFaces(face_cascade, recognizer, CASCADE_PATH, RESULT_OUTPUT_PATH, img_paths, labels)
train_faces.run()

recognizer.read('../results/trained_result.yml')
labels = {v: k for k, v in train_faces.label_ids.items()}

cap = cv.VideoCapture(0)
cap.set(cv.CAP_PROP_FRAME_WIDTH, 1280)
cap.set(cv.CAP_PROP_FRAME_HEIGHT, 720)

face_recognition = FaceRecognition(cap, CASCADE_PATH, face_cascade, recognizer, labels).run()