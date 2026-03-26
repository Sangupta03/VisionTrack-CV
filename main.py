import cv2

from feature_tracking import FeatureTracker
from motion_estimation import MotionEstimator
from trajectory import Trajectory
from object_detection import ObjectDetector
from motion_detection import MotionDetector


def main():
    cap = cv2.VideoCapture("videos/input.mp4")

    tracker = FeatureTracker()
    motion = MotionEstimator()
    traj = Trajectory()
    detector = ObjectDetector()
    motion_detector = MotionDetector()

    prev_frame = None
    prev_kp = None
    prev_des = None

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        # Motion mask
        fgmask = motion_detector.detect(frame)
        cv2.imshow("Motion Mask", fgmask)

        # Object detection
        detections = detector.detect(frame)

        # Feature detection
        kp, des = tracker.detect(frame)

        if prev_frame is not None:
            matches = tracker.match(prev_des, des)

            R, t = motion.estimate(prev_kp, kp, matches)

            if t is not None:
                traj_img = traj.update(t)
                cv2.imshow("Trajectory", traj_img)

            match_img = tracker.draw_matches(prev_frame, prev_kp, frame, kp, matches)
            cv2.imshow("Feature Matching", match_img)

        # Draw objects with motion label
        for (label, conf, (x1, y1, x2, y2)) in detections:
            moving = motion_detector.is_moving((x1, y1, x2, y2), fgmask)

            if moving:
                color = (0, 0, 255)
                text = f"{label} (Moving)"
            else:
                color = (0, 255, 0)
                text = f"{label} (Static)"

            cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
            cv2.putText(frame, text, (x1, y1 - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

        prev_frame = frame
        prev_kp = kp
        prev_des = des

        cv2.imshow("Final Output", frame)

        if cv2.waitKey(1) & 0xFF == 27:
            break

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()