# YOLO11 TensorRT Real-Time Detection

This repository contains two Python applications (`app1.py` and `app2.py`) for real-time object detection using **YOLO11** accelerated with **TensorRT**.  
Both applications use OpenCV for video capture and rendering, and Ultralytics YOLO for inference.  

---

## 📌 Overview

### **`app1.py`** – Real-Time Webcam Detection
- Captures frames from your **webcam**.
- Runs YOLO11 model (TensorRT engine `.engine` file).
- Displays detections in **real-time** with bounding boxes and labels.
- Shows **FPS** and **detection count** overlays.
- Supports interactive keyboard controls.

### **`app2.py`** – Video File Detection & Saving
- Loads frames from a **video file** (`video.mp4` by default).
- Performs YOLO11 inference on each frame.
- Displays results in a window and **saves the annotated video** to `output.mp4`.
- Loops the input video when it finishes.
- Shows system info: PyTorch version, CUDA availability, and GPU count.
- Supports interactive keyboard controls.

🎥 **Demo Video (app2.py result):**  
[Watch on YouTube](https://youtube.com/shorts/1mshRvPiv7k?feature=share)

---

## ⚡ Features

- YOLO11 TensorRT accelerated inference.
- Full-screen or windowed display.
- Dynamic confidence threshold adjustment.
- Bounding boxes with class labels and confidence scores.
- FPS and detection count overlays.
- Output video saving (`app2.py`).

---

## 🔧 Requirements

- Python 3.8+
- git clone https://github.com/mohammed3120/yolo11_tensorRT_repo.git
- python3 -m venv venv
- source venv/bin/activate
- pip install -r requirements.txt


