import cv2 as cv


def end(msg, exit_code=0):
    print(msg)
    exit(exit_code)


class FaceRecognition:
    def __init__(self, cap, cascade_path, cascade_classifier, face_recognizer, labels):
        self.cap = cap
        self.cascade_path = cascade_path
        self.cascade_classifier = cascade_classifier
        self.face_recognizer = face_recognizer
        self.labels = labels

    def run(self):
        if not self.cap.isOpened():
            end("Unable to open camera. Exiting...", -1)
        else:
            while self.cap.isOpened():
                ret, frame = self.cap.read()
                frame = cv.flip(frame, 1)
                if not ret:
                    end("Cannot retrieve frame. Exiting", -1)

                gray_frame = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)

                faces = self.cascade_classifier.detectMultiScale(
                    image=gray_frame,
                    scaleFactor=1.05,
                    minNeighbors=3,
                    minSize=(100, 000)
                )

                for (x, y, w, h) in faces:
                    id, conf = self.face_recognizer.predict(gray_frame[y:y + h, x:x + w])
                    if conf >= 30 and conf <= 80:
                        print(f'Confidence: ({self.labels[id]}) - {conf}')
                        cv.putText(img=frame,
                                   text=self.labels[id].capitalize(), org=(x, y - 5),
                                   fontFace=cv.FONT_HERSHEY_SIMPLEX,
                                   fontScale=1,
                                   thickness=2,
                                   color=(0, 0, 255),
                                   lineType=cv.LINE_AA)
                        cv.rectangle(
                            img=frame,
                            pt1=(x, y),
                            pt2=(x + w, y + h),
                            color=(0, 255, 0),
                            thickness=2,
                            lineType=cv.LINE_AA
                        )

                cv.imshow("Webcam recording", frame)

                k = cv.waitKey(20) & 0xFF
                if k == ord('q') or k == 27:
                    break

            # Release resource
            self.cap.release()
            cv.destroyAllWindows()
