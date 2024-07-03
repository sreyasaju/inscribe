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
                background-color: #171e30;
            }
            QTextEdit {
                background-color: #131c31;
                color: #d4d6ff;
                border: 4px solid #5c6bff;
                border-radius: 15px;
                font: 22pt 'Trebuchet MS';

            }
            QPushButton#start_stop_button {
                background-color: #7587ff;
                color: #1f283e;
                font: 16pt 'Trebuchet MS';
                border: none;
                border-radius: 10px;
                margin: 5px 2px;
                padding: 10px 50px;
            }
            QPushButton#start_stop_button:hover {
                background-color: #8796ff;
            }

            QPushButton#exit_button {
                background-color: #8292ff;
                color: #1f283e;
                font: 16pt 'Trebuchet MS';
                border: none;
                border-radius: 10px;
                margin: 5px 2px;
                padding: 10px 50px;
            }
            QPushButton#exit_button:hover {
                background-color: #98a4ff;
            }

            QLabel {
                background-color: #1e1e2e;
                border: 1px solid #6272a4;
            }

            start
        """)
        self.video_running = True
        self.initUI()

    def initUI(self):
        self.central_widget = QWidget(self)
        self.setCentralWidget(self.central_widget)

        # layouts
        self.main_layout = QHBoxLayout()
        self.video_layout = QVBoxLayout()
        self.text_layout = QVBoxLayout()

        # video frame label
        self.video_label = QLabel(self)
        self.video_label.setFixedSize(640, 480)
        self.video_layout.addWidget(self.video_label, alignment=Qt.AlignCenter)

        # text_field
        self.text_field = QTextEdit(self)
        self.text_field.setReadOnly(True)
        self.text_layout.addWidget(self.text_field)

        # buttons
        self.start_stop_button = QPushButton("Stop Video", self)
        self.start_stop_button.setObjectName("start_stop_button")
        self.start_stop_button.setFixedSize(250, 60)
        self.start_stop_button.clicked.connect(self.toggle_video_feed)
        self.text_layout.addWidget(self.start_stop_button)

        self.exit_button = QPushButton("Exit", self)
        self.exit_button.setObjectName("exit_button")
        self.exit_button.setFixedSize(250, 60)
        self.exit_button.clicked.connect(self.close)
        self.text_layout.addWidget(self.exit_button)

        # adding the  layouts to the main layout
        self.main_layout.addLayout(self.video_layout)
        self.main_layout.addLayout(self.text_layout)

        self.central_widget.setLayout(self.main_layout)

        # Video capture
        self.cap = cv2.VideoCapture(0)

        # setting timer for video feed...
        self.timer = QTimer(self)
        self.timer.timeout.connect(self.update_video_feed)
        self.timer.start(10)

    def recognize_text(self, frame):
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        results = reader.readtext(gray)
        recognized_text = ""

        for (bbox, text, _) in results:
            # bounding box and label
            pts = np.array(bbox, dtype=np.int32)
            cv2.polylines(frame, [pts], isClosed=True, color=(0, 255, 0), thickness=2)
            cv2.putText(frame, text, (pts[0][0], pts[0][1] - 10), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 5)

            recognized_text += text + ' '

        self.text_field.clear()
        self.text_field.append(recognized_text.strip())

        return frame

    def update_video_feed(self):
        ret, frame = self.cap.read()
        if ret:
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)  # Convert frame to RGB for displaying in PyQt5

            # Process frame to recognize text and draw bounding boxes
            frame_with_boxes = self.recognize_text(frame)

            # converting frame to QImage
            h, w, ch = frame_with_boxes.shape
            bytes_per_line = ch * w
            convert_to_Qt_format = QImage(frame_with_boxes.data, w, h, bytes_per_line, QImage.Format_RGB888)
            p = convert_to_Qt_format.scaled(640, 480, Qt.KeepAspectRatio)

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

