

<br>
<h1 align="center"> inscribe</h1>
<p align="center"> Transform handwriting to text on video input </p>
<br>

This Python enables real-time handwriting recognition from a webcam feed using EasyOCR and OpenCV. It detects and interprets handwritten text in each frame, providing visual feedback by drawing bounding boxes around recognized text and displaying the identified text.
## Features

- **Real-time Video Feed:** Displays video from the webcam with recognized text overlaid.
- **Handwriting Recognition:** Uses EasyOCR to detect and recognize text from the video feed.
- **GUI Interface:** Built using PyQt5 for a user-friendly interface with buttons to start/stop the video feed and exit the application.

## Requirements

- Python 3.12
- PyQt5
- OpenCV (cv2)
- EasyOCR

## Installation

Clone the repository:
   ```bash
   git clone https://github.com/sreyasaju/inscribe.git
   cd inscribe
   ```
Install dependencies:
```bash
pip install -r requirements.txt
```

## Usage
1. Run the application:
   - On Linux distros / MacOS, run `python3 main.py`
   - On Windows, run `python main.py`
2. Position the webcam to capture handwritten text.
3. View real-time recognition results on the displayed text field.

## Interface Controls:

- Click Start Video button to begin capturing from the webcam.
- Click Stop Video button to pause the video feed.
- Click Exit button to close the application.

> Add screenshot!

## Dependencies

The project uses the [EasyOCR](https://github.com/JaidedAI/EasyOCR) library for handwriting recognition along with [OpenCV](https://github.com/opencv/opencv) and PyQt for video processing and the ui respectively.

## License
This project is licensed under the MIT License.



```
MIT License
Copyright (c) 2024 Sreya Saju

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
```

Copyright &copy; 2024 [Sreya Saju](https://github.com/sreyasaju)


