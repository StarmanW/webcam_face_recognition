import cv2 as cv
import sys

CASCADE_PATH = '../cascade/data/haarcascade_frontalface_default.xml'
faceCascade = cv.CascadeClassifier(CASCADE_PATH)

cap = cv.VideoCapture(0)
cap.set(cv.CAP_PROP_FRAME_HEIGHT, 720)
cap.set(cv.CAP_PROP_FRAME_WIDTH, 1280)

while True:
    # Capture frame-by-frame
    ret, frame = cap.read()

    # Get grayscale image
    gray = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)

    faces = faceCascade.detectMultiScale(
        gray,
        scaleFactor=1.2,
        minNeighbors=5,
        minSize=(200, 200),
    )

    # Draw a rectangle around the faces
    for (x, y, w, h) in faces:
        cv.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)
        cv.putText(frame, "Person", (x, y - 5), cv.FONT_HERSHEY_SIMPLEX, 1, (0,0,255), 2)

    # Display the resulting frame
    cv.imshow('Video', frame)

    if cv.waitKey(1) & 0xFF == ord('q'):
        break

# When everything is done, release the capture
cap.release()
cv.destroyAllWindows()