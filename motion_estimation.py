import cv2
import numpy as np

class MotionEstimator:
    def __init__(self):
        self.K = np.array([[718, 0, 607],
                           [0, 718, 185],
                           [0, 0, 1]])

    def estimate(self, kp1, kp2, matches):
        if len(matches) < 8:
            return None, None

        pts1 = np.float32([kp1[m.queryIdx].pt for m in matches])
        pts2 = np.float32([kp2[m.trainIdx].pt for m in matches])

        E, _ = cv2.findEssentialMat(pts1, pts2, self.K, method=cv2.RANSAC)

        if E is None:
            return None, None

        _, R, t, _ = cv2.recoverPose(E, pts1, pts2, self.K)
        return R, t