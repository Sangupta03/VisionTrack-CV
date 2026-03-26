import cv2

class MotionDetector:
    def __init__(self):
        self.fgbg = cv2.createBackgroundSubtractorMOG2(history=500, varThreshold=50)

    def detect(self, frame):
        fgmask = self.fgbg.apply(frame)
        fgmask = cv2.medianBlur(fgmask, 5)
        return fgmask

    def is_moving(self, box, fgmask):
        x1, y1, x2, y2 = box

        h, w = fgmask.shape
        x1, y1 = max(0, x1), max(0, y1)
        x2, y2 = min(w, x2), min(h, y2)

        roi = fgmask[y1:y2, x1:x2]

        if roi.size == 0:
            return False

        return cv2.countNonZero(roi) > 500