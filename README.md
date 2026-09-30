\# Real-Time Facial Expression Detection Using Deep Learning



A real-time computer vision application that uses a webcam to detect a person's facial expression and display the predicted expression on the screen.



\## Features



\- Real-time webcam detection

\- Automatic face detection

\- Facial expression recognition

\- Displays the predicted expression and confidence

\- Supports 7 facial expressions:

&#x20; - Angry

&#x20; - Disgust

&#x20; - Fear

&#x20; - Happy

&#x20; - Sad

&#x20; - Surprise

&#x20; - Neutral

\- Uses a pretrained deep learning facial-expression model

\- Built with Python, TensorFlow and OpenCV



\## How It Works



The application follows this pipeline:



Webcam

↓

Face Detection

↓

Face Cropping

↓

Grayscale Conversion

↓

48 × 48 Image

↓

Deep Learning Model

↓

Expression Prediction

↓

Expression + Confidence on Screen



\## Technologies Used



\- Python

\- TensorFlow

\- Keras

\- OpenCV

\- NumPy

\- FER-2013 dataset

\- Pretrained facial-expression model



\## Project Structure



```text

real-time-emotion-detection/

│

├── dataset/

│   └── FER-2013 dataset

│

├── models/

│   └── facial\_expression\_model\_weights.h5

│

├── results/

│

├── src/

│   ├── train.py

│   ├── evaluate.py

│   └── realtime.py

│

├── .gitignore

├── requirements.txt

└── README.md

