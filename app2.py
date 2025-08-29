import cv2
import numpy as np
from ultralytics import YOLO
import time
import torch

class YOLODetector:
    def __init__(self, model_path):
        self.model = YOLO(model_path)
        self.cap = None
        self.out = None  # for saving video
        self.is_fullscreen = True
        self.show_fps = True
        self.show_detections = True
        self.confidence_threshold = 0.5

    def initialize_camera(self, video_path="video.mp4"):
        """Initialize video file"""
        self.cap = cv2.VideoCapture(video_path)
        if not self.cap.isOpened():
            print(f"Error: Could not open video file {video_path}")
            return False

        # Setup video writer for output.mp4 (only one loop will be written)
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        fps = int(self.cap.get(cv2.CAP_PROP_FPS))
        width = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        self.out = cv2.VideoWriter("output.mp4", fourcc, fps, (width, height))
        return True

    def create_window(self):
        """Create full-screen window"""
        cv2.namedWindow("YOLO11 TensorRT Detection", cv2.WND_PROP_FULLSCREEN)
        cv2.setWindowProperty("YOLO11 TensorRT Detection", cv2.WND_PROP_FULLSCREEN, 
                             cv2.WINDOW_FULLSCREEN if self.is_fullscreen else cv2.WINDOW_NORMAL)

    def toggle_fullscreen(self):
        """Toggle fullscreen mode"""
        self.is_fullscreen = not self.is_fullscreen
        self.create_window()

    def process_frame(self, frame):
        """Process frame with YOLO model"""
        results = self.model(frame, verbose=False, conf=self.confidence_threshold)
        return results[0] if results else None

    def draw_detections(self, frame, result):
        """Draw detection results on frame"""
        if result is None or result.boxes is None:
            return frame

        boxes = result.boxes.xyxy.cpu().numpy()
        classes = result.boxes.cls.cpu().numpy()
        confidences = result.boxes.conf.cpu().numpy()
        class_names = result.names

        for i, (box, cls, conf) in enumerate(zip(boxes, classes, confidences)):
            x1, y1, x2, y2 = map(int, box)

            # Generate color based on class
            color = self.get_color(int(cls))

            # Draw bounding box
            cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)

            # Draw label
            label = f"{class_names[int(cls)]} {conf:.2f}"
            self.draw_label(frame, label, x1, y1, color)

        return frame

    def get_color(self, class_id):
        """Generate consistent color for each class"""
        colors = [
            (0, 255, 0), (0, 0, 255), (255, 0, 0), 
            (255, 255, 0), (0, 255, 255), (255, 0, 255),
            (255, 165, 0), (128, 0, 128), (0, 128, 128)
        ]
        return colors[class_id % len(colors)]

    def draw_label(self, frame, text, x, y, color):
        """Draw text label with background"""
        (label_width, label_height), baseline = cv2.getTextSize(
            text, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2
        )

        # Draw background
        cv2.rectangle(
            frame,
            (x, y - label_height - 10),
            (x + label_width, y),
            color,
            -1
        )

        # Draw text
        cv2.putText(
            frame,
            text,
            (x, y - 5),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            (0, 0, 0),
            2
        )

    def draw_info(self, frame, fps, detection_count):
        """Draw information overlay"""
        if self.show_fps:
            cv2.putText(frame, f"FPS: {fps:.1f}", (10, 30), 
                       cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

        if self.show_detections:
            cv2.putText(frame, f"Detections: {detection_count}", (10, 70), 
                       cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        if self.confidence_threshold:
            cv2.putText(frame, f"confidence threshold: {self.confidence_threshold}", (10, 110), 
                       cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

        # Draw help text
        help_text = "Press: F - Toggle Fullscreen, Q - Quit, C - Change Confidence"
        cv2.putText(frame, help_text, (10, frame.shape[0] - 10), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)

    def run(self, video_path="video.mp4"):
        """Main detection loop"""
        if not self.initialize_camera(video_path):
            return

        self.create_window()

        prev_time = 0
        fps = 0

        print("Starting YOLO11 TensorRT Video Detection (loop mode)")
        print("Controls:")
        print("  F - Toggle Fullscreen")
        print("  Q - Quit")
        print("  C - Change Confidence Threshold")
        print("  +/- - Adjust Confidence")

        while True:
            ret, frame = self.cap.read()
            if not ret:
                # Restart video from beginning when it ends
                self.cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
                continue

            # Process frame
            result = self.process_frame(frame)

            # Draw detections
            if self.show_detections and result is not None:
                frame = self.draw_detections(frame, result)
                detection_count = len(result.boxes) if result.boxes else 0
            else:
                detection_count = 0

            # Calculate FPS
            current_time = time.time()
            fps = 1 / (current_time - prev_time) if prev_time != 0 else 0
            prev_time = current_time

            # Draw info overlay
            self.draw_info(frame, fps, detection_count)

            # Display frame
            cv2.imshow("YOLO11 TensorRT Detection", frame)

            # Save frame to output video (only one full loop)
            self.out.write(frame)

            # Handle keyboard input
            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                break
            elif key == ord('f'):
                self.toggle_fullscreen()
            elif key == ord('c'):
                thresholds = [0.3, 0.5, 0.7, 0.9]
                current_idx = thresholds.index(self.confidence_threshold) if self.confidence_threshold in thresholds else 0
                self.confidence_threshold = thresholds[(current_idx + 1) % len(thresholds)]
                print(f"Confidence threshold set to: {self.confidence_threshold}")
            elif key == ord('+'):
                self.confidence_threshold = min(0.9, self.confidence_threshold + 0.1)
                print(f"Confidence threshold: {self.confidence_threshold:.1f}")
            elif key == ord('-'):
                self.confidence_threshold = max(0.1, self.confidence_threshold - 0.1)
                print(f"Confidence threshold: {self.confidence_threshold:.1f}")

        # Cleanup
        self.cap.release()
        self.out.release()
        cv2.destroyAllWindows()

# Main execution
if __name__ == "__main__":
    print(f"PyTorch version: {torch.__version__}")
    print(f"CUDA available: {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        print(f"CUDA version: {torch.version.cuda}")
        print(f"Number of GPUs: {torch.cuda.device_count()}")

    detector = YOLODetector("yolo11n.engine")
    detector.run("video.mp4")
