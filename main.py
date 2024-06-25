import cv2
import easyocr
import sys
from PyQt5.QtWidgets import QApplication, QMainWindow, QLabel, QTextEdit, QPushButton, QVBoxLayout, QWidget, QHBoxLayout
from PyQt5.QtGui import QImage, QPixmap, QPainter, QPen, QFont
from PyQt5.QtCore import QTimer, Qt, QRect
from PIL import Image
import numpy as np

# Initialize EasyOCR
reader = easyocr.Reader(['en'])


class HandwritingRecognitionApp(QMainWindow):
    def __init__(self):
        super().__init__()

        self.setWindowTitle("AI Handwriting Recognition")
        self.setGeometry(100, 100, 1200, 600)
        self.setStyleSheet("""
            QMainWindow {
                background-color: #1e1e2e;
            }
            QTextEdit {
                background-color: #282a36;
                color: #f8f8f2;
                border: 1px solid #6272a4;
                font: 22pt 'Trebuchet MS';
            }
            QPushButton {
                background-color: #44475a;
                color: #f8f8f2;
                font: 12pt 'Helvetica';
                padding: 10px;
                border: none;
                border-radius: 5px;
            }
            QPushButton:hover {
                background-color: #6272a4;
            }
            QLabel {
                background-color: #1e1e2e;
                border: 1px solid #6272a4;
            }
        """)

        self.video_running = True
        self.initUI()

    def initUI(self):
        # Main widget
        self.central_widget = QWidget(self)
        self.setCentralWidget(self.central_widget)

        # Layouts
        self.main_layout = QHBoxLayout()
        self.video_layout = QVBoxLayout()
        self.text_layout = QVBoxLayout()

        # Video feed label
        self.video_label = QLabel(self)
        self.video_label.setFixedSize(640, 480)
        self.video_layout.addWidget(self.video_label, alignment=Qt.AlignCenter)

        # Recognized text field
        self.text_field = QTextEdit(self)
        self.text_field.setReadOnly(True)
        self.text_layout.addWidget(self.text_field)

        # Buttons
        self.start_stop_button = QPushButton("Stop Video", self)
        self.start_stop_button.clicked.connect(self.toggle_video_feed)
        self.text_layout.addWidget(self.start_stop_button)

        self.exit_button = QPushButton("Exit", self)
        self.exit_button.clicked.connect(self.close)
        self.text_layout.addWidget(self.exit_button)

        # Add layouts to the main layout
        self.main_layout.addLayout(self.video_layout)
        self.main_layout.addLayout(self.text_layout)

        # Set main layout
        self.central_widget.setLayout(self.main_layout)

        # Video capture
        self.cap = cv2.VideoCapture(0)

        # Timer for video feed
        self.timer = QTimer(self)
        self.timer.timeout.connect(self.update_video_feed)
        self.timer.start(10)

    def recognize_text(self, frame):
        # Convert frame to grayscale
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        # Use EasyOCR to recognize text
        results = reader.readtext(gray)

        # Update recognized text in GUI
        recognized_text = ""
        for (bbox, text, _) in results:
            # Draw bounding box and label
            pts = np.array(bbox, dtype=np.int32)
            cv2.polylines(frame, [pts], isClosed=True, color=(0, 255, 0), thickness=2)

            # Display the recognized text on the bounding box
            cv2.putText(frame, text, (pts[0][0], pts[0][1] - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

            recognized_text += text + ' '

        # Display recognized text in QTextEdit
        self.text_field.clear()
        self.text_field.append(recognized_text.strip())

        return frame

    def update_video_feed(self):
        ret, frame = self.cap.read()
        if ret:
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)  # Convert frame to RGB for displaying in PyQt5

            # Process frame to recognize text and draw bounding boxes
            frame_with_boxes = self.recognize_text(frame)

            # Convert frame to QImage
            h, w, ch = frame_with_boxes.shape
            bytes_per_line = ch * w
            convert_to_Qt_format = QImage(frame_with_boxes.data, w, h, bytes_per_line, QImage.Format_RGB888)
            p = convert_to_Qt_format.scaled(640, 480, Qt.KeepAspectRatio)

            # Display the frame on the QLabel
            self.video_label.setPixmap(QPixmap.fromImage(p))

    def toggle_video_feed(self):
        self.video_running = not self.video_running
        if self.video_running:
            self.timer.start(10)
            self.start_stop_button.setText("Stop Video")
        else:
            self.timer.stop()
            self.start_stop_button.setText("Start Video")

    def closeEvent(self, event):
        self.cap.release()
        cv2.destroyAllWindows()
        event.accept()


# Run the application
if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = HandwritingRecognitionApp()
    window.show()
    sys.exit(app.exec_())
