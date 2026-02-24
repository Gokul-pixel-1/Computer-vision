 Traffic Object Detection using YOLOv11

Project Overview

This project implements a real-time traffic object detection system using the YOLO (You Only Look Once) deep learning model.
The model is trained to detect common road objects such as:

* Cars
* Traffic lights
* Pedestrians
* Cones

The system can identify these objects in images and draw bounding boxes around them with confidence scores.

 Objectives

*Build a custom object detection model using YOLO
*Annotate a traffic dataset using Roboflow
*Train the model on Google Colab with GPU acceleration

*Evaluate performance using precision, recall, and mAP metrics
*Test the trained model on new images

Tools & Technologies Used

*Python
*YOLOv11 (Ultralytics)
*Google Colab (GPU – Tesla T4)
*Roboflow (Dataset labeling & export)
*PyTorch
*Matplotlib

Dataset

The dataset was created and managed using Roboflow.

Classes:

1. Car
2. Traffic Light
3. Pedestrian
4. Cone

Dataset Details:

*Annotated images exported in YOLO format
*Includes train / validation / test splits
*Bounding boxes manually verified for accuracy

 Model Training:

*Training was performed on Google Colab GPU using Ultralytics YOLO.

Training Configuration:

*Model: YOLOv11s
*Image size: 640x640
*Epochs: 80
*Batch size: 16
*Optimizer: Default Ultralytics optimizer

 Model Performance

*Metric	Score

Precision	0.79 (79%)
Recall	0.86 (86%)
mAP@50	0.88 (88%)
mAP@50–95	0.82 (82%)


Interpretation:

Precision 79% → Most detected objects are correct

Recall 86% → Model detects most real objects present

mAP@50 88% → Strong bounding box accuracy

mAP@50–95 82% → Reliable performance under strict evaluation

Example Output:

The trained model successfully detects:

*Cars in road scenes
*Traffic lights in intersections
*Pedestrians in walking areas
*Safety cones in construction zones

Each detection includes:

*Bounding box
*Class label
*Confidence score

 How to Run the Project

1️) Open Google Colab Notebook

Mount Drive:

from google.colab import drive
drive.mount('/content/drive')

2️) Install YOLO

!pip install ultralytics

3️) Load trained model

from ultralytics import YOLO
model = YOLO("best.pt")

4️) Run prediction on image

model.predict(source="image.jpg", conf=0.5, save=True)

Project Structure

Traffic-Detection-YOLO/
│
├── dataset/
│   ├── train/
│   ├── valid/
│   └── test/
│
├── runs/
│   └── detect/
│
├── best.pt
├── notebook.ipynb
└── README.md

Future Improvements

*Add more training images for better generalization

*Train larger YOLO model (YOLOv11m/l)

*Enable real-time webcam detection

*Deploy as a web application

*Convert to TensorRT for faster inference



Author:

GOKUL S



