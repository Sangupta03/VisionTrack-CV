import cv2

class FeatureTracker:
    def __init__(self):
        self.orb = cv2.ORB_create(2000)
        self.bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)

    def detect(self, frame):
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        kp, des = self.orb.detectAndCompute(gray, None)
        return kp, des

    def match(self, des1, des2):
        if des1 is None or des2 is None:
            return []
        matches = self.bf.match(des1, des2)
        matches = sorted(matches, key=lambda x: x.distance)
        return matches

    def draw_matches(self, frame1, kp1, frame2, kp2, matches):
        return cv2.drawMatches(frame1, kp1, frame2, kp2, matches[:50], None)