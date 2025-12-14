#Real-Time  Sign Language (ASL) Recognition using CNN and deeplearning
This project implements a real-time American Sign Language (ASL) alphabet recognition system using Computer Vision and Deep Learning. It leverages a Convolutional Neural Network (CNN) trained on ASL hand gestures to detect and recognize 29 signs (A–Z + space, nothing, del) through a live webcam feed. Recognized signs are automatically converted into text to form full sentences for enhanced human-computer interaction.

🔧 Tech Stack
Python, TensorFlow/Keras

OpenCV for video processing

CNN for image classification

Real-time prediction and sentence formation logic

📌 Features
Real-time gesture recognition via webcam

Supports full ASL alphabet and control signs (space, nothing, delete)

Sentence builder with prediction stability logic

ROI-based hand segmentation

Confidence filtering to ensure reliable predictions

🧠 Model Input
Input: 64×64 RGB image of hand gesture

Output: Predicted ASL letter with confidence score

🚀 How to Run
Clone the repository

Ensure CNNmodel.h5 (trained model) is present

Run the main script:

bash
Copy
Edit
python RIOlinopenCV.py
Show ASL gestures within the ROI box to start translating into text

Press 'q' to quit
## 📥 Download Dataset

To download the ASL Alphabet dataset from Kaggle:

1. Place your `kaggle.json` API key in `~/.kaggle/` (or same directory).
2. Run:

```bash
pip install kaggle
python download_dataset.py
note:✅ Best Practice: Add a Script to Download from Kaggle
Create a kaggle.json API key
Go to your Kaggle account:
https://www.kaggle.com/account

Scroll to API section → Click “Create New API Token”

It downloads a file called kaggle.json
 dataset link:
 https://www.kaggle.com/datasets/grassknoted/asl-alphabet
