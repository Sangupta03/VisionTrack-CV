# VisionTrack — CV Motion Tracking Pipeline

A real-time computer vision pipeline combining feature tracking, visual odometry, object detection, and motion segmentation.

---

## Features

- ORB feature detection & matching
- Visual odometry via Essential Matrix decomposition
- MobileNet SSD object detection (car, person, bicycle, etc.)
- MOG2 background subtraction for moving object labelling
- 2D camera trajectory visualization

---

## Setup

```bash
pip install -r requirements.txt
```

Download the MobileNet SSD model files into `models/`:

- `deploy.prototxt` — https://github.com/chuanqi305/MobileNet-SSD  
- `mobilenet_iter_73000.caffemodel` — same repo  

Place your video at:

```bash
videos/input.mp4
```

---

## Camera Calibration (Important)

The default intrinsic matrix `K` is calibrated for the *KITTI dataset* and will give wrong odometry results for other cameras.

To calibrate your own camera:

```python
# Print checkerboard pattern, take 15–20 photos, then run:
# https://docs.opencv.org/4.x/dc/dbb/tutorial_py_calibration.html
```

Then update `CAMERA_K` in `main.py` with your calibrated values.

---

## Run

```bash
python main.py
```

Press *ESC* to exit.

---

## Windows

| Window            | Description                                      |
|------------------|--------------------------------------------------|
| Final Output     | Detected objects labelled Moving/Static          |
| Feature Matching | ORB keypoint matches between frames              |
| Motion Mask      | MOG2 foreground mask                             |
| Trajectory       | Estimated camera path (top-down view)            |
