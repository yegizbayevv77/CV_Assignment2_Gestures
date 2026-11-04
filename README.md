# Hand Gesture Recognition: peace / stop / thumbs_up (CV Assignment 2)

Hand gesture classifier built on a self-collected dataset (525 phone photos, 3 classes, 175 per class).
Assignment 2 extends the Assignment 1 baseline with a SOTA hand detector (MediaPipe Hand Landmarker),
which crops the hand before classification, and adds Grad-CAM explanations and a web demo.

## Pipeline

1. Frames are extracted from phone videos (35 frames per video, center-cropped to 256x256).
2. The split is done by video (1 test video and 1 validation video per class), not by frame.
3. MediaPipe Hand Landmarker finds the hand and crops a square region around it (224x224).
   If no hand is found, the whole image is used.
4. A simple CNN is trained on the crops (with augmentation).
5. Grad-CAM shows which image regions the model uses for its prediction.
6. A Gradio app runs upload/webcam -> prediction -> Grad-CAM, and shows the latency.

## Results (test set, 105 images, mean over 5 seeds)

| Model | Accuracy (mean ± std) | Macro F1 (mean ± std) |
|---|---|---|
| A1 baseline, full image | 0.893 ± 0.072 | 0.892 ± 0.073 |
| A2, hand crop (MediaPipe) | 0.924 ± 0.021 | 0.923 ± 0.021 |

The difference is within the seed-to-seed variation, so it is not a statistically clear improvement.
Latency per image: CNN ~16 ms, MediaPipe ~28 ms (Colab CPU/GPU).

## Repository contents

- `CV_Assignment2_Gestures.ipynb`: full notebook (Assignment 1 pipeline + Assignment 2 additions)
- `app.py`: Gradio demo (same prediction and Grad-CAM code as the notebook)
- `a2_crop_model.keras`: trained crop model (copy it from the Google Drive project folder, see below)
- `requirements.txt`: Python packages
- `outputs/`: figures and result tables from the experiments

## How to run the demo

1. Install the packages:
   ```bash
   pip install -r requirements.txt
   ```
2. Download the MediaPipe hand model (about 7.5 MB) into the repository folder:
   ```bash
   wget -O hand_landmarker.task https://storage.googleapis.com/mediapipe-models/hand_landmarker/hand_landmarker/float16/latest/hand_landmarker.task
   ```
3. Make sure `a2_crop_model.keras` is in the same folder as `app.py`.
4. Start the app:
   ```bash
   python app.py
   ```
5. Open the local address printed in the terminal (usually http://127.0.0.1:7860) and upload an image or use the webcam.

## How to reproduce the experiments

1. Open `CV_Assignment2_Gestures.ipynb` in Google Colab with a T4 GPU.
2. Put the dataset videos in `cv_assignment/videos/<class_name>/` in Google Drive (folders `peace`, `stop`, `thumbs_up`).
3. Run all cells in order. The notebook mounts Drive, extracts frames, trains the models and saves the figures and the crop model.

Dataset: [TODO: add Google Drive link to dataset.zip]

## Demo video

[TODO: add Google Drive link to the demo video]

## Limitations

- All photos come from one person and a small number of backgrounds (5 videos per class).
- The model still relies partly on the background. Grad-CAM shows this on some errors, for example
  when a peace sign is classified as thumbs_up with a textured background.
- Hand detection fails on some frames (train 24%, validation 33%, test 8%). For these frames the full image is used.
- Results come from 5 random seeds on one small test set and should be read as rough estimates.

## Ethics and consent

All images were taken by the author of this project, of their own hands. No other people are in the dataset.
No synthetic or generated media is used.

## References

- Zhang, F. et al. (2020). MediaPipe Hands: On-device Real-time Hand Tracking. arXiv:2006.10214.
- Sandler, M. et al. (2018). MobileNetV2: Inverted Residuals and Linear Bottlenecks (Assignment 1 transfer-learning experiment).
- Selvaraju, R. R. et al. (2017). Grad-CAM: Visual Explanations from Deep Networks via Gradient-based Localization.
