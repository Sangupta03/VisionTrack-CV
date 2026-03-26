import numpy as np
import cv2

class Trajectory:
    def __init__(self):
        self.traj = np.zeros((600, 600, 3), dtype=np.uint8)
        self.position = np.zeros((3, 1))

    def update(self, t):
        self.position += t

        x, z = int(self.position[0]), int(self.position[2])

        draw_x = x + 300
        draw_z = z + 300

        cv2.circle(self.traj, (draw_x, draw_z), 2, (0, 255, 0), -1)

        return self.traj