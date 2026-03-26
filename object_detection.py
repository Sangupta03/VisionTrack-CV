import cv2
import numpy as np

class ObjectDetector:
    def __init__(self):
        self.net = cv2.dnn.readNetFromCaffe(
            "models/deploy.prototxt",
            "models/mobilenet_iter_73000.caffemodel"
        )

        self.classes = ["background", "aeroplane", "bicycle", "bird", "boat",
                        "bottle", "bus", "car", "cat", "chair", "cow",
                        "diningtable", "dog", "horse", "motorbike", "person",
                        "pottedplant", "sheep", "sofa", "train", "tvmonitor"]

    def detect(self, frame):
        h, w = frame.shape[:2]

        blob = cv2.dnn.blobFromImage(frame, 0.007843, (300, 300), 127.5)
        self.net.setInput(blob)
        detections = self.net.forward()

        results = []

        for i in range(detections.shape[2]):
            confidence = detections[0, 0, i, 2]

            if confidence > 0.5:
                idx = int(detections[0, 0, i, 1])

                box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
                (x1, y1, x2, y2) = box.astype("int")

                label = self.classes[idx]
                results.append((label, confidence, (x1, y1, x2, y2)))

        return results