📹 Vision Smart Tracker with Interactive Learning

This project is an AI-powered object tracking and classification system using OpenCV, PyTorch, and Tkinter GUI. It supports real-time video analysis, motion detection, and interactive model training. The system can distinguish between people, vehicles, shadows, and unknown objects with online learning capabilities.
🔍 Key Features

    Real-time object detection and tracking using CSRT tracker

    AI-based classification (person, vehicle, shadow, unknown)

    Built-in motion detection and filtering

    GUI interface to monitor frames and adjust zoom/motion settings

    Interactive training to correct detections and manually annotate new samples

    Automatic saving of frames and dataset for further training

    Dynamic learning with confidence-weighted updates

🧠 Classifier Classes

    0: Shadow / Noise

    1: Person

    2: Vehicle

    3: Unknown

🛠 Installation
1. Clone the repository

git clone https://github.com/superbodik/VisionASCII.git
cd vision-smart-tracker

2. Create and activate a virtual environment (optional but recommended)

python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

3. Install dependencies

pip install -r requirements.txt

    🔧 Make sure you have a working camera and Python 3.8 or newer.

🚀 Run the application

python main.py

The GUI will launch, showing a live video feed. You can:

    Adjust zoom and motion threshold

    See tracked objects with ID, class, speed, and confidence

    Enable saving of snapshots for dataset generation

    Use the training window to correct or add manual annotations

📦 Updated requirements.txt

opencv-contrib-python
numpy
torch
torchvision
Pillow

    If you're using GPU acceleration, consider installing torch with CUDA support:
    https://pytorch.org/get-started/locally/ 

📁 Folder Structure

    main.py - Main application and video loop

    camera.py - Camera and zoom functions

    gui.py - Tkinter-based user interface

    neural_classifier.py - Interactive annotation and training interface

    interactive_training.py - Neural network, learner and shadow classifier

    ascii_converter.py - Converts frames to ASCII (optional feature)

    dataset.py - PyTorch dataset for image-ASCII pairs

    utils.py - Frame/ascii saving utilities

    smart_tracker_model.pth - Pretrained model (optional)

💡 Example Use Cases

    Security systems with intelligent camera monitoring

    Smart surveillance with adaptive learning

    Robotics and AI environments with dynamic scene understanding
